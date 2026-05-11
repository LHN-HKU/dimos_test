// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C++ codec for the dimos `sensor_msgs.RawLidarScan` LCM payload.
//
// Hand-rolled binary layout (must match
// `dimos/msgs/sensor_msgs/RawLidarScan.py` byte for byte):
//
//   magic        : 4 bytes  "RLS\x01"
//   ts_sec       : i32      ROS seconds
//   ts_nsec      : u32      ROS nanoseconds
//   frame_len    : u16
//   frame_id     : frame_len bytes (UTF-8)
//   point_count  : u32
//   points       : point_count × {
//                    x f32, y f32, z f32,
//                    reflectivity u16,
//                    offset_time_ns u32,
//                    line u16,
//                    tag u8,
//                  }  = 21 bytes/point (packed, no trailing pad)
//
// Little-endian on all targets we ship to (x86_64 / aarch64 little-endian).

#ifndef DIMOS_HARDWARE_LIDAR_COMMON_RAW_LIDAR_SCAN_HPP_
#define DIMOS_HARDWARE_LIDAR_COMMON_RAW_LIDAR_SCAN_HPP_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dimos::raw_lidar {

constexpr char MAGIC[4] = {'R', 'L', 'S', '\x01'};

#pragma pack(push, 1)
struct Point {
    float x;
    float y;
    float z;
    uint16_t reflectivity;
    uint32_t offset_time_ns;
    uint16_t line;
    uint8_t tag;
};
#pragma pack(pop)
static_assert(sizeof(Point) == 21, "RawLidarScan Point must be 21 bytes (matches Python wire layout)");

struct ScanHeader {
    int32_t ts_sec;
    uint32_t ts_nsec;
    std::string frame_id;
};

// Encode a complete RawLidarScan payload into `out`.
inline void encode(std::vector<uint8_t>& out, const ScanHeader& header, const Point* points, uint32_t point_count) {
    const uint16_t frame_len = static_cast<uint16_t>(header.frame_id.size());
    out.resize(
        sizeof(MAGIC)
        + sizeof(int32_t)
        + sizeof(uint32_t)
        + sizeof(uint16_t)
        + frame_len
        + sizeof(uint32_t)
        + static_cast<size_t>(point_count) * sizeof(Point)
    );
    uint8_t* w = out.data();
    std::memcpy(w, MAGIC, sizeof(MAGIC));               w += sizeof(MAGIC);
    std::memcpy(w, &header.ts_sec, sizeof(int32_t));    w += sizeof(int32_t);
    std::memcpy(w, &header.ts_nsec, sizeof(uint32_t));  w += sizeof(uint32_t);
    std::memcpy(w, &frame_len, sizeof(uint16_t));       w += sizeof(uint16_t);
    if (frame_len > 0) {
        std::memcpy(w, header.frame_id.data(), frame_len);
        w += frame_len;
    }
    std::memcpy(w, &point_count, sizeof(uint32_t));     w += sizeof(uint32_t);
    if (point_count > 0) {
        std::memcpy(w, points, static_cast<size_t>(point_count) * sizeof(Point));
    }
}

// Decode a RawLidarScan payload. Returns false on malformed input.
// `points_out` and `header_out` are filled on success.
inline bool decode(const uint8_t* data, size_t len, ScanHeader& header_out, std::vector<Point>& points_out) {
    const uint8_t* r = data;
    const uint8_t* end = data + len;
    const size_t fixed_pre = sizeof(MAGIC) + sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint16_t);
    if (len < fixed_pre) return false;
    if (std::memcmp(r, MAGIC, sizeof(MAGIC)) != 0) return false;
    r += sizeof(MAGIC);
    std::memcpy(&header_out.ts_sec, r, sizeof(int32_t));   r += sizeof(int32_t);
    std::memcpy(&header_out.ts_nsec, r, sizeof(uint32_t)); r += sizeof(uint32_t);
    uint16_t frame_len;
    std::memcpy(&frame_len, r, sizeof(uint16_t));          r += sizeof(uint16_t);
    if (r + frame_len + sizeof(uint32_t) > end) return false;
    header_out.frame_id.assign(reinterpret_cast<const char*>(r), frame_len);
    r += frame_len;
    uint32_t point_count;
    std::memcpy(&point_count, r, sizeof(uint32_t));        r += sizeof(uint32_t);
    const size_t points_bytes = static_cast<size_t>(point_count) * sizeof(Point);
    if (r + points_bytes > end) return false;
    points_out.resize(point_count);
    if (point_count > 0) {
        std::memcpy(points_out.data(), r, points_bytes);
    }
    return true;
}

}  // namespace dimos::raw_lidar

#endif  // DIMOS_HARDWARE_LIDAR_COMMON_RAW_LIDAR_SCAN_HPP_
