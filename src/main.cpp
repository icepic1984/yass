#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <glfw.hpp>
#include <GLFW/glfw3.h>

// #define GLM_FORCE_RADIANS
// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
// #include <glm/mat4x4.hpp>
// #include <glm/vec4.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

const bool enableValidationLayer = true;

// clang-format off
const std::vector<const char*> validationLayers{
    "VK_LAYER_LUNARG_standard_validation",
    "VK_LAYER_LUNARG_parameter_validation",
    "VK_LAYER_LUNARG_object_tracker",
    "VK_LAYER_LUNARG_core_validation"

};
// clang-format on

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pCallback)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

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
    std::copy(extensions, extensions + extensionCount, result.begin());
    if (enableValidationLayer)
        result.push_back("VK_EXT_debug_utils");
    return result;
}

bool checkValidationLayer()
{
    auto layerProperties = vk::enumerateInstanceLayerProperties();
    bool available = false;

    for (const auto& validationLayer : validationLayers) {
        auto found = std::find_if(
            layerProperties.begin(), layerProperties.end(),
            [&validationLayer](const vk::LayerProperties& layer) {
                if (std::strcmp(validationLayer, layer.layerName) == 0)
                    return true;
                return false;
            });

        if (found == layerProperties.end())
            return false;
    }
    return true;
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
            for (const auto& iter : requiredExtensionNames)
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
    return result.size() == requiredExtensionNames.size();
}

bool checkDeviceExtensionSupport(const vk::PhysicalDevice& physicalDevice)
{
    auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
    std::vector<vk::ExtensionProperties> result;

    std::copy_if(extensions.begin(), extensions.end(),
                 std::back_inserter(result),
                 [](const vk::ExtensionProperties& property) {
                     for (const auto& iter : deviceExtensions) {
                         if (strcmp(property.extensionName, iter) == 0)
                             return true;
                     }
                     return false;
                 });
    return result.size() == deviceExtensions.size();
}

bool isDeviceSuitable(const vk::PhysicalDevice& device)
{
    auto properties = device.getProperties();
    auto features = device.getFeatures();
    // Test if device is a dedicated graphic card and supports a
    // geometry shader.
    return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
           features.geometryShader && checkDeviceExtensionSupport(device);
}

vk::PhysicalDevice pickDevice(const vk::UniqueInstance& instance)
{
    // Pick first device which meets the requirements (aka is a
    // dedicated graphic card and has a geometry shader)
    auto devices = instance->enumeratePhysicalDevices();
    for (const auto& iter : instance->enumeratePhysicalDevices()) {
        if (isDeviceSuitable(iter))
            return iter;
    }
    throw std::runtime_error(
        "No graphic cards with vulkan support found on system.");
    return vk::PhysicalDevice();
}

void printQueueProperties(const vk::PhysicalDevice& device)
{
    std::cout << "Queue properties of: " << device.getProperties().deviceName
              << std::endl;

    auto queues = device.getQueueFamilyProperties();
    for (const auto& iter : queues) {
        std::cout << "Properties ("
                  << static_cast<unsigned int>(iter.queueFlags)
                  << "): Queue count = " << iter.queueCount << " Familie = ";
        if (iter.queueFlags & vk::QueueFlagBits::eProtected)
            std::cout << "Protected ";
        if (iter.queueFlags & vk::QueueFlagBits::eGraphics)
            std::cout << "Graphic ";
        if (iter.queueFlags & vk::QueueFlagBits::eTransfer)
            std::cout << "Transfer ";
        if (iter.queueFlags & vk::QueueFlagBits::eCompute)
            std::cout << "Compute ";
        if (iter.queueFlags & vk::QueueFlagBits::eSparseBinding)
            std::cout << "SparseBinding ";
        std::cout << std::endl;
    }
}

std::optional<uint32_t> findQueueFamilies(const vk::PhysicalDevice& device)
{

    auto queues = device.getQueueFamilyProperties();
    for (std::size_t i = 0; i < queues.size(); ++i) {
        if (queues[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return static_cast<uint32_t>(i);
    }
    return std::nullopt;
}

vk::UniqueInstance
createInstance(const std::vector<const char*>& requiredExtensions)
{
    vk::ApplicationInfo appInfo{"Test"};

    // Setup InstanceCreateInfo struct which is used to configure
    // the vulkan instance.
    vk::InstanceCreateInfo createInfo{
        vk::InstanceCreateFlags(),
        &appInfo,
        0,
        nullptr,
        static_cast<uint32_t>(requiredExtensions.size()),
        requiredExtensions.data()};

    if (checkValidationLayer() && enableValidationLayer) {
        createInfo.setEnabledLayerCount(
            static_cast<uint32_t>(validationLayers.size()));
        createInfo.setPpEnabledLayerNames(validationLayers.data());
    }

    // Create vulkan instance
    return vk::createInstanceUnique(createInfo);
}

vk::UniqueDevice createDevice(const vk::PhysicalDevice& physicalDevice,
                              uint32_t queueFamilyIndex)
{
    // Create device from physical device with one queue.
    float priority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &priority);
    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), 1,
                                          &deviceQueueCreateInfo);
    if (checkValidationLayer() && enableValidationLayer) {
        deviceCreateInfo.setEnabledLayerCount(
            static_cast<uint32_t>(validationLayers.size()));
        deviceCreateInfo.setPpEnabledLayerNames(validationLayers.data());
    }
    return physicalDevice.createDeviceUnique(deviceCreateInfo);
}

auto createDebugMessenger(const vk::UniqueInstance& instance,
                          const vk::DispatchLoaderDynamic& dldi)
{
    vk::DebugUtilsMessengerCreateInfoEXT debugInfo(
        vk::DebugUtilsMessengerCreateFlagsEXT(),
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
        &debugCallback, nullptr);

    // Initialize dynamic dispatch
    // Register debug callback for layer validation
    return instance->createDebugUtilsMessengerEXTUnique(debugInfo, nullptr,
                                                        dldi);
}

int main()
{
    glfwInit();

    auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    auto requiredExtensions = getRequiredExtensionsForGlfw();

    if (meetExtensionRequirements(availableExtensions, requiredExtensions)) {
        // Create vulkan instance
        std::cout << "Create Instance" << std::endl;
        auto instance = createInstance(requiredExtensions);
        vk::DispatchLoaderDynamic dldi(instance.get());
        std::cout << "Create DebugMessagenger" << std::endl;
        auto messanger = createDebugMessenger(instance, dldi);
        std::cout << "Create window" << std::endl;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto window = glfw::createWindow(800, 600, "test");
        auto surface = window.createWindowSurface(instance);
        auto physicalDevice = pickDevice(instance);
        auto queueFamilyIndex = findQueueFamilies(physicalDevice);

        if (!queueFamilyIndex.has_value()) {
            throw std::runtime_error("No suitable queue family found");
        }
        // printQueueProperties(physicalDevice);

        // Test if queue of device supports presentation
        if (!physicalDevice.getSurfaceSupportKHR(*queueFamilyIndex,
                                                 surface.get())) {
            throw std::runtime_error("Queue does not support presentation");
        }

        std::cout << "Device " << std::endl;

        auto device = createDevice(physicalDevice, *queueFamilyIndex);
        auto queue = device->getQueue(*queueFamilyIndex, 0);
    }

    std::cout << "Glfw extensions" << std::endl;

    //  initVulkan();
    // uint32_t extensionCount = 0;

    // auto result = vk::enumerateInstanceLayerProperties();
    // std::cout << result.size() << std::endl;

    // glfwInit();

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
