#include "camera.h"

#include <plog/Log.h>

#include <chrono>

namespace {

float get_depth_scale(const std::optional<rs2::depth_sensor>& sensor) {
    if (sensor.has_value()) {
        return sensor.value().get_depth_scale();
    } else {
        LOG_WARNING << "Failed to get depth scale; defaulting to 0.001\n";
        return 0.001;
    }
}

template <typename T>
std::optional<T> get_sensor(const rs2::pipeline_profile& profile) {
    try {
        return profile.get_device().first<T>();
    } catch (rs2::error) {
        LOG_WARNING << "Failed to get sensor typeid:"
                    << std::string{typeid(T).name()};
        return std::nullopt;
    }
}

inline cv::Mat frame_to_mat(auto frame, int type) {
    return cv::Mat(frame.get_height(), frame.get_width(), type,
                   (void*)frame.get_data());
}

cv::Mat normalize_u16_to_rgb(cv::Mat&& mat) {
    double minv, maxv;
    cv::minMaxLoc(mat, &minv, &maxv);

    // avoid degenerate case (uniform data)
    if (maxv - minv < 1e-3) {
        maxv = minv + 1.0;
    }

    cv::Mat f16;
    mat.convertTo(f16, CV_32F);
    f16 = (f16 - static_cast<float>(minv)) *
          (255.0f / static_cast<float>(maxv - minv));

    cv::Mat u8;
    f16.convertTo(u8, CV_8U);
    return u8;

    cv::Mat rgb;
    cv::cvtColor(u8, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
}

inline cv::Mat gray_to_rgb(cv::Mat&& mat) {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    return rgb;
}

}  // namespace

namespace vision {

Frames::Frames(rs2::video_frame color_frame, rs2::depth_frame depth_frame,
               rs2::video_frame depth_color_frame, rs2::video_frame ir_frame)
    : _color_frame(std::move(color_frame)),
      _depth_frame(std::move(depth_frame)),
      _depth_color_frame(std::move(depth_color_frame)),
      _ir_frame(std::move(ir_frame)) {
    // _color_bgr = frame_to_mat(_color_frame, CV_8UC3);

    _color_bgr =
        cv::Mat(_color_frame.get_height(), _color_frame.get_width(), CV_8UC3,
                (void*)_color_frame.get_data(), cv::Mat::AUTO_STEP);
    if (!_color_bgr.isContinuous()) {
        _color_bgr = _color_bgr.clone();
    }
    _depth_z16 = frame_to_mat(_depth_frame, CV_16U);
    _depth_rgb = frame_to_mat(_depth_color_frame, CV_8UC3);
    _ir_y8 = gray_to_rgb(frame_to_mat(_ir_frame, CV_8UC1));
}

const cv::Mat& Frames::color() const { return _color_bgr; }

const cv::Mat& Frames::color_depth() const { return _depth_rgb; }

const cv::Mat& Frames::depth() const { return _depth_z16; }

const cv::Mat& Frames::ir() const { return _ir_y8; }

float Frames::get_distance(int x, int y) const {
    return _depth_frame.get_distance(x, y);
}

Camera::Camera(int width, int height, int fps)
    : _align_to_color(RS2_STREAM_COLOR) {
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8,
                      fps);
    _profile = _pipe.start(cfg);
    _depth_sensor = get_sensor<rs2::depth_sensor>(_profile);
    _depth_scale = get_depth_scale(_depth_sensor);
}

Camera::~Camera() { _pipe.stop(); }

std::optional<Frames> Camera::wait_for_frames() {
    const auto frames = _pipe.wait_for_frames();
    const auto aligned_frames = _align_to_color.process(frames);

    auto color = aligned_frames.get_color_frame();
    auto depth = aligned_frames.get_depth_frame();
    auto ir = aligned_frames.get_infrared_frame();

    if (!color || !depth || !ir) {
        return std::nullopt;
    }

    auto depth_colorized = _colorizer.process(depth);

    return Frames{std::move(color), std::move(depth),
                  std::move(depth_colorized), std::move(ir)};
}

float Camera::depth_scale() const { return _depth_scale; }

std::optional<float> Camera::get_exposure() const {
    return get_option(RS2_OPTION_EXPOSURE);
}

void Camera::set_exposure(float exposure) {
    set_option(RS2_OPTION_EXPOSURE, exposure);
}

std::optional<float> Camera::get_option(rs2_option option) const {
    if (_depth_sensor.has_value()) {
        try {
            return _depth_sensor.value().get_option(option);
        } catch (const rs2::error& e) {
            LOG_ERROR << "Failed to get" +
                             std::string{rs2_option_to_string(option)} + ": " +
                             e.what();
            return std::nullopt;
        }
    } else {
        LOG_ERROR << "Failed to get" +
                         std::string{rs2_option_to_string(option)} +
                         ". No valid depth sensor was found";
        return std::nullopt;
    }
}

void Camera::set_option(rs2_option option, float value) {
    if (_depth_sensor.has_value()) {
        try {
            _depth_sensor.value().set_option(option, value);
        } catch (const rs2::error& e) {
            LOG_ERROR << "Failed to set" +
                             std::string{rs2_option_to_string(option)} + ": " +
                             e.what();
            return;
        }
    } else {
        LOG_ERROR << "Failed to set" +
                         std::string{rs2_option_to_string(option)} +
                         " . No valid depth sensor was found";
    }
}

}  // namespace vision
