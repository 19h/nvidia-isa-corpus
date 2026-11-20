/* Run a single-invocation Vulkan SC workload from an offline pipeline cache. */
#include <vulkan/vulkan_sc.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { \
    VkResult result = (call); \
    if (result != VK_SUCCESS) { \
        fprintf(stderr, "%s failed (%d)\n", #call, result); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void fail(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *read_cache(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END))
        fail("Could not open pipeline cache");
    long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET))
        fail("Invalid pipeline cache size");
    void *data = malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, file) != (size_t)length)
        fail("Could not read pipeline cache");
    fclose(file);
    *size = (size_t)length;
    return data;
}

int main(int argc, char **argv)
{
    if ((argc != 4 && argc != 5) || strlen(argv[2]) != 2 * VK_UUID_SIZE ||
        strspn(argv[2], "0123456789abcdefABCDEF") != 2 * VK_UUID_SIZE) {
        fprintf(stderr, "Usage: %s cache.bin pipeline-uuid-hex shader-entry [input]\n", argv[0]);
        return EXIT_FAILURE;
    }
    enum { WORDS = 64 };
    /* Some ports use the same descriptor for static, zero-initialized scratch storage. */
    const VkDeviceSize buffer_size = 4 * 1024 * 1024;
    const VkDeviceSize pipeline_size = 4 * 1024 * 1024;

    VkPipelineCacheCreateInfo cache_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .flags = VK_PIPELINE_CACHE_CREATE_READ_ONLY_BIT | VK_PIPELINE_CACHE_CREATE_USE_APPLICATION_STORAGE_BIT,
    };
    void *cache_data = read_cache(argv[1], &cache_info.initialDataSize);
    cache_info.pInitialData = cache_data;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan SC corpus runner",
        .apiVersion = VKSC_API_VERSION_1_0,
    };
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance instance;
    CHECK(vkCreateInstance(&instance_info, NULL, &instance));

    uint32_t count = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (!count)
        fail("No Vulkan SC physical devices");
    VkPhysicalDevice *physical_devices = calloc(count, sizeof(*physical_devices));
    if (!physical_devices)
        fail("Could not allocate the physical device list");
    CHECK(vkEnumeratePhysicalDevices(instance, &count, physical_devices));
    VkPhysicalDevice physical = physical_devices[0];
    free(physical_devices);

    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
    if (!families)
        fail("Could not allocate the queue family list");
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
    uint32_t family = 0;
    while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    free(families);
    if (family == count)
        fail("No compute queue family");

    VkPhysicalDeviceVulkan11Features supported11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features supported12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &supported11,
    };
    VkPhysicalDeviceFeatures2 supported = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &supported12,
    };
    vkGetPhysicalDeviceFeatures2(physical, &supported);

    VkPipelinePoolSize pool_size = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_POOL_SIZE,
        .poolEntrySize = pipeline_size,
        .poolEntryCount = 1,
    };
    VkDeviceObjectReservationCreateInfo reservations = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_OBJECT_RESERVATION_CREATE_INFO,
        .pipelineCacheCreateInfoCount = 1, .pPipelineCacheCreateInfos = &cache_info,
        .pipelinePoolSizeCount = 1, .pPipelinePoolSizes = &pool_size,
        .commandBufferRequestCount = 1, .bufferRequestCount = 1,
        .deviceMemoryRequestCount = 1, .pipelineCacheRequestCount = 1,
        .pipelineLayoutRequestCount = 1, .computePipelineRequestCount = 1,
        .descriptorSetLayoutRequestCount = 1, .descriptorPoolRequestCount = 1,
        .descriptorSetRequestCount = 1, .descriptorSetLayoutBindingLimit = 1,
        .commandPoolRequestCount = 1,
    };
    VkPhysicalDeviceVulkan11Features features11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &reservations,
        .variablePointers = supported11.variablePointers,
        .variablePointersStorageBuffer = supported11.variablePointersStorageBuffer,
        .storageBuffer16BitAccess = supported11.storageBuffer16BitAccess,
    };
    VkPhysicalDeviceVulkan12Features features12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &features11,
        .shaderInt8 = supported12.shaderInt8,
        .storageBuffer8BitAccess = supported12.storageBuffer8BitAccess,
    };
    VkPhysicalDeviceFeatures features = {
        .shaderFloat64 = supported.features.shaderFloat64,
        .shaderInt64 = supported.features.shaderInt64,
        .shaderInt16 = supported.features.shaderInt16,
    };
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features12,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
        .pEnabledFeatures = &features,
    };
    VkDevice device;
    CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buffer_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer buffer;
    CHECK(vkCreateBuffer(device, &buffer_info, NULL, &buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    uint32_t memory_type = 0;
    const VkMemoryPropertyFlags coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while (memory_type < memory_properties.memoryTypeCount &&
           (!(requirements.memoryTypeBits & (1u << memory_type)) ||
            (memory_properties.memoryTypes[memory_type].propertyFlags & coherent) != coherent))
        ++memory_type;
    if (memory_type == memory_properties.memoryTypeCount)
        fail("No host-coherent memory type for results");
    VkMemoryAllocateInfo memory_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size, .memoryTypeIndex = memory_type,
    };
    VkDeviceMemory memory;
    CHECK(vkAllocateMemory(device, &memory_info, NULL, &memory));
    CHECK(vkBindBufferMemory(device, buffer, memory, 0));
    void *mapped;
    CHECK(vkMapMemory(device, memory, 0, buffer_size, 0, &mapped));
    memset(mapped, 0, buffer_size);
    if (argc == 5) {
        char *end = NULL;
        unsigned long input = strtoul(argv[4], &end, 10);
        if (end == argv[4] || *end || input > UINT32_MAX)
            fail("Expected a 32-bit unsigned input value");
        ((uint32_t *)mapped)[WORDS - 1] = (uint32_t)input;
    }

    VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    VkDescriptorSetLayout descriptor_layout;
    CHECK(vkCreateDescriptorSetLayout(device, &layout_info, NULL, &descriptor_layout));
    VkDescriptorPoolSize descriptor_count = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &descriptor_count,
    };
    VkDescriptorPool descriptor_pool;
    CHECK(vkCreateDescriptorPool(device, &descriptor_pool_info, NULL, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &descriptor_layout,
    };
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(device, &set_info, &set));
    VkDescriptorBufferInfo buffer_binding = {buffer, 0, buffer_size};
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_binding,
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &descriptor_layout,
    };
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, &layout));
    VkPipelineCache cache;
    CHECK(vkCreatePipelineCache(device, &cache_info, NULL, &cache));
    VkPipelineOfflineCreateInfo offline = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_OFFLINE_CREATE_INFO,
        .matchControl = VK_PIPELINE_MATCH_CONTROL_APPLICATION_UUID_EXACT_MATCH,
        .poolEntrySize = pipeline_size,
    };
    for (size_t i = 0; i < VK_UUID_SIZE; i++) {
        char hex[] = {argv[2][2 * i], argv[2][2 * i + 1], '\0'};
        offline.pipelineIdentifier[i] = (uint8_t)strtoul(hex, NULL, 16);
    }
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = &offline,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = argv[3],
        },
        .layout = layout,
    };
    VkPipeline pipeline;
    CHECK(vkCreateComputePipelines(device, cache, 1, &pipeline_info, NULL, &pipeline));

    VkCommandPoolMemoryReservationCreateInfo command_reservation = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_MEMORY_RESERVATION_CREATE_INFO,
        .commandPoolReservedSize = 1024 * 1024, .commandPoolMaxCommandBuffers = 1,
    };
    VkCommandPoolCreateInfo command_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = &command_reservation, .queueFamilyIndex = family,
    };
    VkCommandPool command_pool;
    CHECK(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool));
    VkCommandBufferAllocateInfo command_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer command;
    CHECK(vkAllocateCommandBuffers(device, &command_info, &command));
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK(vkBeginCommandBuffer(command, &begin));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(command, 1, 1, 1);
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
    CHECK(vkEndCommandBuffer(command));
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &command,
    };
    CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));

    const uint32_t *words = mapped;
    for (unsigned i = 0; i < WORDS; i++)
        printf("%08x%c", words[i], i == WORDS - 1 ? '\n' : ' ');

    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineCache(device, cache, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, NULL);
    vkUnmapMemory(device, memory);
    vkDestroyBuffer(device, buffer, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    free(cache_data);
    return EXIT_SUCCESS;
}
