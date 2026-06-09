#ifndef QEW_FORMAT_H
#define QEW_FORMAT_H

#include <cstdio>
#include <string>

// Tiny printf-into-std::string helper shared by the simulation executables to
// build the HDF5 group keys. The key strings (e.g. "pinning_length=%.3g") MUST
// match the Python scripts' formatting so the same analysis/plot code reads
// either output -- see test_executables.cpp, which pins the documented keys.
namespace qew {

inline std::string fmt(const char *spec, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), spec, v);
    return std::string(buf);
}

}  // namespace qew

#endif  // QEW_FORMAT_H
