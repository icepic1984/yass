#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

std::vector<const char*> getRequiredExtensionsForGlfw()
{
    uint32_t extensionCount = 0;
    const char** extensions = nullptr;
    extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (!extensions) {
        throw std::runtime_error(
            "Could not retrieve list of required vk extensions");
    }
    std::vector<const char*> result(extensionCount);
    for (uint32_t i = 0; i < extensionCount; ++i) {
        result[i] = extensions[i];
    }
    return result;
}

std::vector<vk::ExtensionProperties> filterExtensions(
    const std::vector<vk::ExtensionProperties>& availableExtensions,
    const std::vector<const char*>& requiredExtensionNames)
{
    std::vector<vk::ExtensionProperties> result;
    std::copy_if(
        availableExtensions.begin(), availableExtensions.end(),
        std::back_inserter(result),
        [&requiredExtensionNames](const vk::ExtensionProperties& extension) {
            for (auto& iter : requiredExtensionNames)
                if (std::strcmp(iter, extension.extensionName) == 0)
                    return true;
            return false;
        });
    return result;
}

bool meetExtensionRequirements(
    const std::vector<vk::ExtensionProperties>& availableExtensions,
    const std::vector<const char*>& requiredExtensionNames)
{
    auto result = filterExtensions(availableExtensions, requiredExtensionNames);
    return !result.empty();
}

void initVulkan()
{
    auto instanceExtensionsProperties =
        vk::enumerateInstanceExtensionProperties();
    vk::ApplicationInfo appInfo{"Test"};
    vk::InstanceCreateInfo createInfo;

    for (auto& iter : instanceExtensionsProperties) {
        std::cout << "Specversion: " << iter.specVersion << " "
                  << iter.extensionName << std::endl;
    }

    // createInfo.

    // vk::createInstance(const InstanceCreateInfo &createInfo,
    // Optional<const AllocationCallbacks> allocator = nullptr, const
    // Dispatch &d = Dispatch())
}

int main()
{
    glfwInit();

    auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    auto requiredExtensions = getRequiredExtensionsForGlfw();

    auto filteredExtensions =
        filterExtensions(availableExtensions, requiredExtensions);

    for (auto iter : filteredExtensions) {
        std::cout << "Iter: " << iter.extensionName << " " << iter.specVersion
                  << std::endl;
    }
    if (meetExtensionRequirements(availableExtensions, requiredExtensions)) {
        vk::ApplicationInfo appInfo{"Test"};
        vk::InstanceCreateInfo createInfo{
            vk::InstanceCreateFlags(),
            &appInfo,
            0,
            nullptr,
            static_cast<uint32_t>(requiredExtensions.size()),
            requiredExtensions.data()};

        auto instance = vk::createInstance(createInfo);
    }

    std::cout << "Glfw extensions" << std::endl;
    initVulkan();
    // uint32_t extensionCount = 0;

    // auto result = vk::enumerateInstanceLayerProperties();
    // std::cout << result.size() << std::endl;

    // glfwInit();

    // glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // GLFWwindow* window =
    //     glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);

    // vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
    // nullptr);

    // std::cout << extensionCount << " extensions supported" << std::endl;

    // // glm::mat4 matrix;
    // // glm::vec4 vec;
    // // auto test = matrix * vec;

    // while (!glfwWindowShouldClose(window)) {
    //     glfwPollEvents();
    // }

    // glfwDestroyWindow(window);

    // glfwTerminate();

    return 0;
}
