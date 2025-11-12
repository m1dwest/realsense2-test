#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "vision/detector.h"

inline float get_depth_scale(const rs2::pipeline_profile& profile) {
    auto depth_scale = 0.f;

    const auto sensors = profile.get_device().query_sensors();
    for (const auto& s : sensors) {
        if (auto ds = s.as<rs2::depth_sensor>()) {
            depth_scale = ds.get_depth_scale();
            break;
        }
    }

    if (depth_scale <= 0.f) {
        std::cerr << "Failed to get depth scale; defaulting to 0.001\n";
        depth_scale = 0.001f;
    }

    return depth_scale;
}

inline float get_median_depth(const cv::Mat& depth_z16, const cv::Rect& roi,
                              float depth_scale) {
    cv::Rect clipped = roi & cv::Rect(0, 0, depth_z16.cols, depth_z16.rows);
    if (clipped.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<uint16_t> vals;
    vals.reserve(clipped.area());
    for (int y = clipped.y; y < clipped.y + clipped.height; ++y) {
        const auto* const row = depth_z16.ptr<uint16_t>(y);
        for (int x = clipped.x; x < clipped.x + clipped.width; ++x) {
            const auto d = row[x];
            if (d != 0) {
                vals.push_back(d);
            }
        }
    }

    if (vals.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    return vals[mid] * depth_scale;
}

inline void render(const rs2::pipeline_profile& profile,
                   const std::vector<vision::DetectedObject>& detections,
                   const cv::Mat& color_bgr, const cv::Mat& depth_z16,
                   const cv::Mat& depth_rgb) {
    static const auto surfaces =
        std::vector<const cv::Mat*>{&color_bgr, &depth_rgb};
    static std::size_t surface_index = 0;
    static const auto* surface = surfaces[surface_index];

    auto depth_scale = get_depth_scale(profile);

    for (const auto& d : detections) {
        const cv::Rect& box = d.box;

        float obj_depth_m = get_median_depth(depth_z16, box, depth_scale);
        std::string depth_str =
            std::isnan(obj_depth_m) ? "n/a" : cv::format("%.2fm", obj_depth_m);
        std::string score_str = cv::format("%.2f", d.score);

        // std::optional<cv::Rect> clipped_bottle;
        // if (d.class_name == "bottle") {
        //     clipped_bottle =
        //         box & cv::Rect(0, 0, color_bgr.cols, color_bgr.rows);
        // }

        cv::rectangle(*surface, box, cv::Scalar(30, 119, 252), 2);
        std::string label = d.class_name + " " + depth_str + " " + score_str;
        int base;
        cv::Size tsize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &base);
        int ty = std::max(0, box.y - 8);
        cv::rectangle(*surface,
                      cv::Rect(box.x, ty - tsize.height - 6, tsize.width + 6,
                               tsize.height + 6),
                      cv::Scalar(30, 119, 252), cv::FILLED);
        // if (clipped_bottle.has_value()) {
        //     cv::rectangle(drawing_surface, clipped_bottle.value(),
        //                   cv::Scalar(255, 0, 0), cv::FILLED);
        // }
        cv::putText(*surface, label, cv::Point(box.x + 3, ty - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255),
                    2);
        // std::cout << d.class_name << ": " << depth_str << "\n";
    }

    cv::imshow("Color Image", *surfaces[surface_index]);
    int k = cv::waitKey(1);
    if (k == 27 || k == 'q') exit(0);
    if (k == 'w') {
        ++surface_index;
        if (surface_index >= surfaces.size()) {
            surface_index = 0;
        }
    }
}
