#pragma once

#include <optional>
#include <string>

#include <glad/glad.h>

#include <GLFW/glfw3.h>

namespace gui {

class Application {
   public:
    struct Window {
        int width;
        int height;
        std::string title;
        GLFWwindow* window = nullptr;
        float hiDPIScale;
    };

    struct VideoStream {
        int width;
        int height;
        unsigned texture;
    };

    Application() = default;
    ~Application();

    [[nodiscard]] bool init(int width, int height, std::string title);
    void create_video_stream(int width, int height);
    void update_video_stream(unsigned char* data) const;
    void compose_frame() const;
    bool should_close() const;
    void render() const;
    void input() const;

   private:
    std::optional<Window> _window;
    std::optional<VideoStream> _video_stream;
};

}  // namespace gui
