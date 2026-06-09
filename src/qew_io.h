#ifndef QEW_IO_H
#define QEW_IO_H

// HDF5 output for the three simulations. Header-only (HighFive is header-only)
// and compiled only where HDF5 is available, so it is included by the
// executables and by the HDF5-guarded tests, but never by the core `lib`
// (which does not link HighFive). The dataset KEYS written here are a contract
// with the Python analysis/plot scripts -- test_executables.cpp reads them back
// and asserts the exact strings.

#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "qew_format.h"
#include "qew_runs.h"
#include "qew_types.h"

namespace qew {

inline void write_static(HighFive::File &file, const StaticConfig &c,
                         const std::vector<StaticProfile> &profiles) {
    file.createDataSet("physical_size", static_cast<double>(c.Lx));
    for (const StaticProfile &prof : profiles) {
        const std::string prefix = "driving_force=" + fmt("%.1g", c.driving_force) +
                                   "/pinning_length=" + fmt("%.3g", prof.pinning_length);
        file.createDataSet(prefix + "/pinning_length", prof.pinning_length);
        file.createDataSet(prefix + "/h", prof.h);
    }
}

inline void write_dynamic(HighFive::File &file, const DynamicConfig &c,
                          const DynamicResult &res) {
    file.createDataSet("physical_size", static_cast<double>(c.Lx));
    for (const DynamicSnapshot &snap : res.snapshots) {
        const std::string prefix =
            "timestep=" + fmt("%.6g", snap.t) +
            "/driving_force=" + fmt("%.1g", c.driving_force) +
            "/line_tension=" + fmt("%.3g", res.line_tension);
        file.createDataSet(prefix + "/pinning_length",
                           static_cast<double>(c.pinning_length));
        file.createDataSet(prefix + "/h", snap.h);
    }
}

inline void write_depinning(HighFive::File &file, const DepinningConfig &c,
                            const DepinningResult &res) {
    file.createDataSet("physical_size", static_cast<double>(c.Lx));
    for (const DepinningStep &st : res.steps) {
        if (!st.result.blocked) continue;
        const std::string prefix = "driving_force=" + fmt("%.4g", st.f);
        file.createDataSet(prefix + "/h", st.h);
        file.createDataSet(prefix + "/pinning_length",
                           static_cast<double>(c.pinning_length));
    }
    if (res.f_c_upper_bound >= 0)
        file.createDataSet("f_c_upper_bound",
                           static_cast<double>(res.f_c_upper_bound));
}

}  // namespace qew

#endif  // QEW_IO_H
