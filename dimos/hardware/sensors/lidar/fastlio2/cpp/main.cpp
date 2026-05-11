// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// FAST-LIO2 pure native module for the dimos NativeModule framework.
//
// Takes raw IMU (`sensor_msgs.Imu`) and raw lidar (`sensor_msgs.RawLidarScan`,
// hand-rolled binary payload — see dimos/msgs/sensor_msgs/RawLidarScan.py)
// over LCM and feeds them into FastLio.  Publishes registered world-frame
// point clouds and odometry on LCM.  No hardware SDK linkage — pair with a
// sensor module (e.g. `livox/cpp/mid360_native`) that publishes the raw
// streams.
//
// Usage:
//   ./fastlio2_native \
//       --raw_imu   '/imu#sensor_msgs.Imu' \
//       --raw_lidar '/raw_lidar#sensor_msgs.RawLidarScan' \
//       --lidar     '/lidar#sensor_msgs.PointCloud2' \
//       --odometry  '/odometry#nav_msgs.Odometry' \
//       --config_path /path/to/mid360.yaml \
//       --frame_id world --child_frame_id base_link

#include <lcm/lcm-cpp.hpp>

#include <atomic>
#include <boost/make_shared.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cloud_filter.hpp"
#include "dimos_native_module.hpp"
#include "raw_lidar_scan.hpp"
#include "voxel_map.hpp"

// dimos LCM message headers
#include "geometry_msgs/Quaternion.hpp"
#include "geometry_msgs/Vector3.hpp"
#include "nav_msgs/Odometry.hpp"
#include "sensor_msgs/Imu.hpp"
#include "sensor_msgs/PointCloud2.hpp"
#include "sensor_msgs/PointField.hpp"

// FAST-LIO (header-only core, compiled sources linked via CMake)
#include "fast_lio.hpp"

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{true};
static lcm::LCM* g_lcm = nullptr;
static FastLio* g_fastlio = nullptr;

static std::string g_lidar_topic;
static std::string g_odometry_topic;
static std::string g_map_topic;
static std::string g_frame_id;        // required via --frame_id
static std::string g_child_frame_id;   // required via --child_frame_id

// Initial pose offset (applied to all SLAM outputs)
// Position offset
static double g_init_x = 0.0;
static double g_init_y = 0.0;
static double g_init_z = 0.0;
// Orientation offset as quaternion (identity = no rotation)
static double g_init_qx = 0.0;
static double g_init_qy = 0.0;
static double g_init_qz = 0.0;
static double g_init_qw = 1.0;

// Helper: quaternion multiply (Hamilton product)  q_out = q1 * q2
static void quat_mul(double ax, double ay, double az, double aw,
                     double bx, double by, double bz, double bw,
                     double& ox, double& oy, double& oz, double& ow) {
    ow = aw*bw - ax*bx - ay*by - az*bz;
    ox = aw*bx + ax*bw + ay*bz - az*by;
    oy = aw*by - ax*bz + ay*bw + az*bx;
    oz = aw*bz + ax*by - ay*bx + az*bw;
}

// Helper: rotate a vector by a quaternion  v_out = q * v * q_inv
static void quat_rotate(double qx, double qy, double qz, double qw,
                        double vx, double vy, double vz,
                        double& ox, double& oy, double& oz) {
    // t = 2 * cross(q_xyz, v)
    double tx = 2.0 * (qy*vz - qz*vy);
    double ty = 2.0 * (qz*vx - qx*vz);
    double tz = 2.0 * (qx*vy - qy*vx);
    // v_out = v + qw*t + cross(q_xyz, t)
    ox = vx + qw*tx + (qy*tz - qz*ty);
    oy = vy + qw*ty + (qz*tx - qx*tz);
    oz = vz + qw*tz + (qx*ty - qy*tx);
}

// Check if initial pose is non-identity
static bool has_init_pose() {
    return g_init_x != 0.0 || g_init_y != 0.0 || g_init_z != 0.0 ||
           g_init_qx != 0.0 || g_init_qy != 0.0 || g_init_qz != 0.0 || g_init_qw != 1.0;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using dimos::time_from_seconds;
using dimos::make_header;

// ---------------------------------------------------------------------------
// Publish lidar (world-frame point cloud)
// ---------------------------------------------------------------------------

static void publish_lidar(CloudType::Ptr cloud, double timestamp,
                          const std::string& topic = "") {
    const std::string& chan = topic.empty() ? g_lidar_topic : topic;
    if (!g_lcm || !cloud || cloud->empty() || chan.empty()) return;

    int num_points = static_cast<int>(cloud->size());

    sensor_msgs::PointCloud2 pc;
    pc.header = make_header(g_frame_id, timestamp);
    pc.height = 1;
    pc.width = num_points;
    pc.is_bigendian = 0;
    pc.is_dense = 1;

    // Fields: x, y, z, intensity (float32 each)
    pc.fields_length = 4;
    pc.fields.resize(4);

    auto make_field = [](const std::string& name, int32_t offset) {
        sensor_msgs::PointField f;
        f.name = name;
        f.offset = offset;
        f.datatype = sensor_msgs::PointField::FLOAT32;
        f.count = 1;
        return f;
    };

    pc.fields[0] = make_field("x", 0);
    pc.fields[1] = make_field("y", 4);
    pc.fields[2] = make_field("z", 8);
    pc.fields[3] = make_field("intensity", 12);

    pc.point_step = 16;
    pc.row_step = pc.point_step * num_points;

    pc.data_length = pc.row_step;
    pc.data.resize(pc.data_length);

    // Apply the full init_pose transform (rotation + translation) to point clouds.
    // FAST-LIO's map origin is at the sensor's initial position.  The rotation
    // corrects axis direction (e.g. 180° X for upside-down mount) and the
    // translation shifts the origin so that ground sits at z≈0 (e.g. z=1.2
    // for a sensor mounted 1.2m above ground).  This matches the odometry
    // frame, which also gets the full init_pose applied.
    const bool apply_init_pose = has_init_pose();
    for (int i = 0; i < num_points; ++i) {
        float* dst = reinterpret_cast<float*>(pc.data.data() + i * 16);
        if (apply_init_pose) {
            double rx, ry, rz;
            quat_rotate(g_init_qx, g_init_qy, g_init_qz, g_init_qw,
                        cloud->points[i].x, cloud->points[i].y, cloud->points[i].z,
                        rx, ry, rz);
            dst[0] = static_cast<float>(rx + g_init_x);
            dst[1] = static_cast<float>(ry + g_init_y);
            dst[2] = static_cast<float>(rz + g_init_z);
        } else {
            dst[0] = cloud->points[i].x;
            dst[1] = cloud->points[i].y;
            dst[2] = cloud->points[i].z;
        }
        dst[3] = cloud->points[i].intensity;
    }

    g_lcm->publish(chan, &pc);
}

// ---------------------------------------------------------------------------
// Publish odometry
// ---------------------------------------------------------------------------

static void publish_odometry(const custom_messages::Odometry& odom, double timestamp) {
    if (!g_lcm) return;

    nav_msgs::Odometry msg;
    msg.header = make_header(g_frame_id, timestamp);
    msg.child_frame_id = g_child_frame_id;

    // Pose (apply initial pose offset: p_out = R_init * p_slam + t_init)
    if (has_init_pose()) {
        double rx, ry, rz;
        quat_rotate(g_init_qx, g_init_qy, g_init_qz, g_init_qw,
                    odom.pose.pose.position.x,
                    odom.pose.pose.position.y,
                    odom.pose.pose.position.z,
                    rx, ry, rz);
        msg.pose.pose.position.x = rx + g_init_x;
        msg.pose.pose.position.y = ry + g_init_y;
        msg.pose.pose.position.z = rz + g_init_z;

        double ox, oy, oz, ow;
        quat_mul(g_init_qx, g_init_qy, g_init_qz, g_init_qw,
                 odom.pose.pose.orientation.x,
                 odom.pose.pose.orientation.y,
                 odom.pose.pose.orientation.z,
                 odom.pose.pose.orientation.w,
                 ox, oy, oz, ow);
        msg.pose.pose.orientation.x = ox;
        msg.pose.pose.orientation.y = oy;
        msg.pose.pose.orientation.z = oz;
        msg.pose.pose.orientation.w = ow;
    } else {
        msg.pose.pose.position.x = odom.pose.pose.position.x;
        msg.pose.pose.position.y = odom.pose.pose.position.y;
        msg.pose.pose.position.z = odom.pose.pose.position.z;
        msg.pose.pose.orientation.x = odom.pose.pose.orientation.x;
        msg.pose.pose.orientation.y = odom.pose.pose.orientation.y;
        msg.pose.pose.orientation.z = odom.pose.pose.orientation.z;
        msg.pose.pose.orientation.w = odom.pose.pose.orientation.w;
    }

    // Covariance (fixed-size double[36])
    for (int i = 0; i < 36; ++i) {
        msg.pose.covariance[i] = odom.pose.covariance[i];
    }

    // Twist (zero — FAST-LIO doesn't output velocity directly)
    msg.twist.twist.linear.x = 0;
    msg.twist.twist.linear.y = 0;
    msg.twist.twist.linear.z = 0;
    msg.twist.twist.angular.x = 0;
    msg.twist.twist.angular.y = 0;
    msg.twist.twist.angular.z = 0;
    std::memset(msg.twist.covariance, 0, sizeof(msg.twist.covariance));

    g_lcm->publish(g_odometry_topic, &msg);
}

// ---------------------------------------------------------------------------
// LCM input subscribers
// ---------------------------------------------------------------------------
//
// raw_lidar handler: decodes `sensor_msgs.RawLidarScan` (hand-rolled binary
// payload), builds a CustomMsg, and feeds it to FastLio.
//
// raw_imu handler: decodes `sensor_msgs.Imu` (dimos-lcm) and feeds the
// corresponding custom_messages::Imu directly to FastLio.
//
// LCM's typed subscribe expects member-fn pointers; we wrap the handlers in
// a small struct purely so we can pass them as method pointers.

struct RawHandlers {
    void on_raw_lidar(const lcm::ReceiveBuffer* rbuf, const std::string& chan);
    void on_raw_imu(const lcm::ReceiveBuffer* rbuf, const std::string& chan);
};

void RawHandlers::on_raw_lidar(const lcm::ReceiveBuffer* rbuf, const std::string& /*chan*/) {
    if (!g_running.load() || !g_fastlio || rbuf == nullptr || rbuf->data == nullptr) return;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(rbuf->data);
    dimos::raw_lidar::ScanHeader header;
    std::vector<dimos::raw_lidar::Point> points;
    if (!dimos::raw_lidar::decode(bytes, rbuf->data_size, header, points)) {
        fprintf(stderr, "[fastlio2] raw_lidar: malformed payload (%u bytes), dropping\n",
                rbuf->data_size);
        return;
    }
    if (points.empty()) return;

    const uint64_t frame_ts_ns =
        static_cast<uint64_t>(static_cast<int64_t>(header.ts_sec)) * 1'000'000'000ull
        + header.ts_nsec;

    auto lidar_msg = boost::make_shared<custom_messages::CustomMsg>();
    lidar_msg->header.seq = 0;
    lidar_msg->header.stamp = custom_messages::Time().fromSec(static_cast<double>(frame_ts_ns) / 1e9);
    lidar_msg->header.frame_id = header.frame_id.empty() ? std::string("lidar") : header.frame_id;
    lidar_msg->timebase = frame_ts_ns;
    lidar_msg->lidar_id = 0;
    for (int i = 0; i < 3; i++) lidar_msg->rsvd[i] = 0;
    lidar_msg->point_num = static_cast<uli>(points.size());
    lidar_msg->points.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        auto& cp = lidar_msg->points[i];
        const auto& p = points[i];
        cp.x = static_cast<double>(p.x);
        cp.y = static_cast<double>(p.y);
        cp.z = static_cast<double>(p.z);
        cp.reflectivity = p.reflectivity;
        cp.tag = p.tag;
        cp.line = static_cast<uint8_t>(p.line);
        cp.offset_time = static_cast<uli>(p.offset_time_ns);
    }
    g_fastlio->feed_lidar(lidar_msg);
}

void RawHandlers::on_raw_imu(const lcm::ReceiveBuffer* rbuf, const std::string& /*chan*/) {
    if (!g_running.load() || rbuf == nullptr || !g_fastlio) return;
    sensor_msgs::Imu in;
    if (in.decode(rbuf->data, 0, rbuf->data_size) < 0) {
        fprintf(stderr, "[fastlio2] raw_imu: malformed payload (%u bytes), dropping\n",
                rbuf->data_size);
        return;
    }

    const double ts = static_cast<double>(in.header.stamp.sec) + in.header.stamp.nsec / 1e9;

    auto imu_msg = boost::make_shared<custom_messages::Imu>();
    imu_msg->header.stamp = custom_messages::Time().fromSec(ts);
    imu_msg->header.seq = 0;
    imu_msg->header.frame_id = in.header.frame_id.empty() ? std::string("imu") : in.header.frame_id;

    imu_msg->orientation.x = in.orientation.x;
    imu_msg->orientation.y = in.orientation.y;
    imu_msg->orientation.z = in.orientation.z;
    imu_msg->orientation.w = in.orientation.w;
    for (int j = 0; j < 9; ++j) imu_msg->orientation_covariance[j] = 0.0;

    imu_msg->angular_velocity.x = in.angular_velocity.x;
    imu_msg->angular_velocity.y = in.angular_velocity.y;
    imu_msg->angular_velocity.z = in.angular_velocity.z;
    for (int j = 0; j < 9; ++j) imu_msg->angular_velocity_covariance[j] = 0.0;

    imu_msg->linear_acceleration.x = in.linear_acceleration.x;
    imu_msg->linear_acceleration.y = in.linear_acceleration.y;
    imu_msg->linear_acceleration.z = in.linear_acceleration.z;
    for (int j = 0; j < 9; ++j) imu_msg->linear_acceleration_covariance[j] = 0.0;

    g_fastlio->feed_imu(imu_msg);
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

static void signal_handler(int /*sig*/) {
    g_running.store(false);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    dimos::NativeModule mod(argc, argv);

    // Required: LCM topics for output ports
    g_lidar_topic = mod.has("lidar") ? mod.topic("lidar") : "";
    g_odometry_topic = mod.has("odometry") ? mod.topic("odometry") : "";
    g_map_topic = mod.has("global_map") ? mod.topic("global_map") : "";

    if (g_lidar_topic.empty() && g_odometry_topic.empty()) {
        fprintf(stderr, "Error: at least one of --lidar or --odometry is required\n");
        return 1;
    }

    // FAST-LIO config path
    std::string config_path = mod.arg("config_path", "");
    if (config_path.empty()) {
        fprintf(stderr, "Error: --config_path <path> is required\n");
        return 1;
    }

    // FAST-LIO internal processing rates
    double msr_freq = mod.arg_float("msr_freq", 50.0f);
    double main_freq = mod.arg_float("main_freq", 5000.0f);

    // Required: LCM input topics for raw sensor streams
    std::string raw_imu_topic   = mod.has("raw_imu")   ? mod.topic("raw_imu")   : "";
    std::string raw_lidar_topic = mod.has("raw_lidar") ? mod.topic("raw_lidar") : "";
    if (raw_imu_topic.empty() || raw_lidar_topic.empty()) {
        fprintf(stderr, "Error: --raw_imu and --raw_lidar are both required\n");
        return 1;
    }

    g_frame_id = mod.arg_required("frame_id");
    g_child_frame_id = mod.arg_required("child_frame_id");
    float pointcloud_freq = mod.arg_float("pointcloud_freq", 5.0f);
    float odom_freq = mod.arg_float("odom_freq", 50.0f);
    CloudFilterConfig filter_cfg;
    filter_cfg.voxel_size = mod.arg_float("voxel_size", 0.1f);
    filter_cfg.sor_mean_k = mod.arg_int("sor_mean_k", 50);
    filter_cfg.sor_stddev = mod.arg_float("sor_stddev", 1.0f);
    float map_voxel_size = mod.arg_float("map_voxel_size", 0.1f);
    float map_max_range = mod.arg_float("map_max_range", 100.0f);
    float map_freq = mod.arg_float("map_freq", 0.0f);

    // Initial pose offset [x, y, z, qx, qy, qz, qw]
    {
        std::string init_str = mod.arg("init_pose", "");
        if (!init_str.empty()) {
            double vals[7] = {0, 0, 0, 0, 0, 0, 1};
            int n = 0;
            size_t pos = 0;
            while (pos < init_str.size() && n < 7) {
                size_t comma = init_str.find(',', pos);
                if (comma == std::string::npos) comma = init_str.size();
                vals[n++] = std::stod(init_str.substr(pos, comma - pos));
                pos = comma + 1;
            }
            g_init_x = vals[0]; g_init_y = vals[1]; g_init_z = vals[2];
            g_init_qx = vals[3]; g_init_qy = vals[4]; g_init_qz = vals[5]; g_init_qw = vals[6];
        }
    }

    printf("[fastlio2] Starting pure FAST-LIO2 native module\n");
    if (has_init_pose()) {
        printf("[fastlio2] init_pose: xyz=(%.3f, %.3f, %.3f) quat=(%.4f, %.4f, %.4f, %.4f)\n",
               g_init_x, g_init_y, g_init_z, g_init_qx, g_init_qy, g_init_qz, g_init_qw);
    }
    printf("[fastlio2] raw_imu topic: %s\n", raw_imu_topic.c_str());
    printf("[fastlio2] raw_lidar topic: %s\n", raw_lidar_topic.c_str());
    printf("[fastlio2] lidar topic: %s\n",
           g_lidar_topic.empty() ? "(disabled)" : g_lidar_topic.c_str());
    printf("[fastlio2] odometry topic: %s\n",
           g_odometry_topic.empty() ? "(disabled)" : g_odometry_topic.c_str());
    printf("[fastlio2] global_map topic: %s\n",
           g_map_topic.empty() ? "(disabled)" : g_map_topic.c_str());
    printf("[fastlio2] config: %s\n", config_path.c_str());
    printf("[fastlio2] pointcloud_freq: %.1f Hz  odom_freq: %.1f Hz\n",
           pointcloud_freq, odom_freq);
    printf("[fastlio2] voxel_size: %.3f  sor_mean_k: %d  sor_stddev: %.1f\n",
           filter_cfg.voxel_size, filter_cfg.sor_mean_k, filter_cfg.sor_stddev);
    if (!g_map_topic.empty())
        printf("[fastlio2] map_voxel_size: %.3f  map_max_range: %.1f  map_freq: %.1f Hz\n",
               map_voxel_size, map_max_range, map_freq);

    // Signal handlers
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    // Init LCM
    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "Error: LCM init failed\n");
        return 1;
    }
    g_lcm = &lcm;

    // Init FAST-LIO with config
    printf("[fastlio2] Initializing FAST-LIO...\n");
    FastLio fast_lio(config_path, msr_freq, main_freq);
    g_fastlio = &fast_lio;
    printf("[fastlio2] FAST-LIO initialized.\n");

    // Subscribe to raw sensor streams
    RawHandlers handlers;
    lcm.subscribe(raw_lidar_topic, &RawHandlers::on_raw_lidar, &handlers);
    lcm.subscribe(raw_imu_topic, &RawHandlers::on_raw_imu, &handlers);
    printf("[fastlio2] Subscribed to raw sensor streams, waiting for input...\n");

    const double process_period_ms = 1000.0 / main_freq;

    // Rate limiters for output publishing
    auto pc_interval = std::chrono::microseconds(
        static_cast<int64_t>(1e6 / pointcloud_freq));
    auto odom_interval = std::chrono::microseconds(
        static_cast<int64_t>(1e6 / odom_freq));
    auto last_pc_publish = std::chrono::steady_clock::now();
    auto last_odom_publish = std::chrono::steady_clock::now();

    // Global voxel map (only if map topic is configured AND map_freq > 0)
    std::unique_ptr<VoxelMap> global_map;
    std::chrono::microseconds map_interval{0};
    auto last_map_publish = std::chrono::steady_clock::now();
    if (!g_map_topic.empty() && map_freq > 0.0f) {
        global_map = std::make_unique<VoxelMap>(map_voxel_size, map_max_range);
        map_interval = std::chrono::microseconds(
            static_cast<int64_t>(1e6 / map_freq));
    }

    while (g_running.load()) {
        auto loop_start = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::steady_clock::now();

        // Drain any pending LCM messages (raw_imu / raw_lidar feed FastLio inside the handlers).
        lcm.handleTimeout(0);

        // Run FAST-LIO processing step (high frequency)
        fast_lio.process();

        // Check for new results and accumulate/publish (rate-limited)
        auto pose = fast_lio.get_pose();
        if (!pose.empty() && (pose[0] != 0.0 || pose[1] != 0.0 || pose[2] != 0.0)) {
            double ts = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            auto world_cloud = fast_lio.get_world_cloud();
            if (world_cloud && !world_cloud->empty()) {
                auto filtered = filter_cloud<PointType>(world_cloud, filter_cfg);

                // Per-scan publish at pointcloud_freq
                if (!g_lidar_topic.empty() && now - last_pc_publish >= pc_interval) {
                    publish_lidar(filtered, ts);
                    last_pc_publish = now;
                }

                // Global map: insert, prune, and publish at map_freq
                if (global_map) {
                    global_map->insert<PointType>(filtered);

                    if (now - last_map_publish >= map_interval) {
                        global_map->prune(
                            static_cast<float>(pose[0]),
                            static_cast<float>(pose[1]),
                            static_cast<float>(pose[2]));
                        auto map_cloud = global_map->to_cloud<PointType>();
                        publish_lidar(map_cloud, ts, g_map_topic);
                        last_map_publish = now;
                    }
                }
            }

            // Publish odometry (rate-limited to odom_freq)
            if (!g_odometry_topic.empty() && (now - last_odom_publish >= odom_interval)) {
                publish_odometry(fast_lio.get_odometry(), ts);
                last_odom_publish = now;
            }
        }

        // Rate control (~5kHz processing)
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        if (elapsed_ms < process_period_ms) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<int64_t>((process_period_ms - elapsed_ms) * 1000)));
        }
    }

    // Cleanup
    printf("[fastlio2] Shutting down...\n");
    g_fastlio = nullptr;
    g_lcm = nullptr;

    printf("[fastlio2] Done.\n");
    return 0;
}
