// Unit tests for the executables' command-line parser (executables/cli.h).
// The parser is dependency-free (no Kokkos / qew types), so these tests are
// pure host code -- they pin the contract the three mains rely on: defaults are
// kept when an option is absent, every config field is overridable, negative
// values and comma lists parse, and typos / malformed numbers are rejected.

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "cli.h"  // tests/ adds ../executables to the include path

using qew::cli::Args;

namespace {
// Build an Args from a brace list of string literals, prepending argv[0].
Args make(std::vector<const char *> v) {
    v.insert(v.begin(), "prog");
    return Args(static_cast<int>(v.size()), const_cast<char **>(v.data()));
}
}  // namespace

TEST(Cli, DefaultsKeptWhenAbsent) {
    Args a = make({});
    EXPECT_FALSE(a.help);
    EXPECT_EQ(a.integer("nx", 4096), 4096);
    EXPECT_DOUBLE_EQ(a.dbl("ftol", 1e-6), 1e-6);
    EXPECT_EQ(a.u64("seed", 1), 1u);
    EXPECT_EQ(a.str("model", "arclength"), "arclength");
    const std::vector<double> def = {0.01, 0.1, 1.0};
    EXPECT_EQ(a.dbl_list("pinning_lengths", def), def);
    EXPECT_NO_THROW(a.require_all_used());
}

TEST(Cli, OverridesBothSpellings) {
    Args a = make({"--nx", "8192", "--ftol=1e-8", "--model", "linear"});
    EXPECT_EQ(a.integer("nx", 4096), 8192);
    EXPECT_DOUBLE_EQ(a.dbl("ftol", 1e-6), 1e-8);
    EXPECT_EQ(a.str("model", "arclength"), "linear");
    EXPECT_NO_THROW(a.require_all_used());
}

TEST(Cli, NegativeValueIsNotMistakenForFlag) {
    Args a = make({"--driving_force", "-1.5"});
    EXPECT_DOUBLE_EQ(a.dbl("driving_force", 0.0), -1.5);
    EXPECT_NO_THROW(a.require_all_used());
}

TEST(Cli, CommaSeparatedList) {
    Args a = make({"--pinning_lengths", "0.1,1,10,100"});
    const std::vector<double> got = a.dbl_list("pinning_lengths", {});
    ASSERT_EQ(got.size(), 4u);
    EXPECT_DOUBLE_EQ(got[0], 0.1);
    EXPECT_DOUBLE_EQ(got[3], 100.0);
}

TEST(Cli, HelpFlag) {
    EXPECT_TRUE(make({"--help"}).help);
    EXPECT_TRUE(make({"-h"}).help);
}

TEST(Cli, UnknownOptionRejected) {
    Args a = make({"--bogus", "3"});
    a.integer("nx", 1);  // does not touch --bogus
    EXPECT_THROW(a.require_all_used(), std::runtime_error);
}

TEST(Cli, MalformedNumberRejected) {
    Args a = make({"--nx", "abc"});
    EXPECT_THROW(a.integer("nx", 1), std::runtime_error);
    Args b = make({"--ftol", "1.0x"});  // trailing junk
    EXPECT_THROW(b.dbl("ftol", 1.0), std::runtime_error);
}

TEST(Cli, MissingValueRejected) {
    EXPECT_THROW(make({"--nx"}), std::runtime_error);
}

TEST(Cli, NonOptionTokenRejected) {
    EXPECT_THROW(make({"positional"}), std::runtime_error);
}
