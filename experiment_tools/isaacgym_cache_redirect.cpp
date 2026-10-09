// Narrow process-local redirection for Isaac Gym Preview 4 on Linux.
// Overrides only two exported carb::gym path helpers; never changes HOME,
// getpwuid, account records, system libraries, or the simulation algorithm.
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
using PathFunction = std::string (*)();

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "DexGrasp IsaacGym cache redirect: %s\n", message);
    std::fflush(stderr);
    _exit(78);
}

std::string configured_path(const char* value, const char* allowed) {
    if (std::strcmp(value, allowed) != 0) {
        fail("configured path must exactly match the approved E-drive directory");
    }
    char* canonical = ::realpath(value, nullptr);
    if (canonical == nullptr) {
        fail("approved directory must already exist before simulation starts");
    }
    const bool matches = std::strcmp(canonical, allowed) == 0;
    std::free(canonical);
    if (!matches) {
        fail("approved directory resolves outside its expected E-drive location");
    }
    return std::string(value);
}

PathFunction next_function(const char* symbol) {
    ::dlerror();
    void* function = ::dlsym(RTLD_NEXT, symbol);
    if (::dlerror() != nullptr || function == nullptr) {
        fail("unconfigured pass-through could not resolve the original Gym function");
    }
    return reinterpret_cast<PathFunction>(function);
}
}

namespace carb { namespace gym {
__attribute__((visibility("default"))) std::string getHomeDirectory() {
    const char* value = std::getenv("DEXGRASP_SIM_HOME");
    if (value != nullptr) {
        return configured_path(value, "/mnt/e/DexGraspResearch/cache/isaacgym_home");
    }
    static PathFunction original = next_function("_ZN4carb3gym16getHomeDirectoryB5cxx11Ev");
    return original();
}

__attribute__((visibility("default"))) std::string getTempDirectory() {
    const char* value = std::getenv("DEXGRASP_SIM_TEMP");
    if (value != nullptr) {
        return configured_path(value, "/mnt/e/DexGraspResearch/temp");
    }
    static PathFunction original = next_function("_ZN4carb3gym16getTempDirectoryB5cxx11Ev");
    return original();
}
}}
