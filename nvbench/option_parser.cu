#include <nvbench/option_parser.cuh>

#include <nvbench/benchmark_manager.cuh>
#include <nvbench/range.cuh>

#include <fmt/format.h>

#include <cassert>
#include <charconv>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{

//==============================================================================
// helpers types for using std::string_view with std::regex
using sv_citer          = std::string_view::const_iterator;
using sv_match          = std::match_results<sv_citer>;
using sv_submatch       = std::sub_match<sv_citer>;
using sv_regex_iterator = std::regex_iterator<sv_citer>;
std::string_view submatch_to_sv(const sv_submatch &in)
{
  // This will be much easier in C++20, but this string_view constructor is
  // painfully absent until then:
  // return {in.first, in.second};

  // C++17 version:
  if (in.first == in.second)
  {
    return {};
  }

  // We have to use the (ptr, len) ctor
  return {&*in.first, static_cast<std::size_t>(in.length())};
}
//==============================================================================

template <typename T>
void parse(std::string_view input, T &val)
{
  // std::from_chars requires const char *, not iterators, grumble grumble
  auto [_, err] = std::from_chars(&*input.cbegin(), &*input.cend(), val);
  if (err != std::errc())
  {
    throw std::runtime_error(fmt::format("{}:{}: Error parsing value from "
                                         "string '{}'",
                                         __FILE__,
                                         __LINE__,
                                         input));
  }
}

void parse(std::string_view input, std::string &val) { val = input; }

// Parses a list of values "<val1>, <val2>, <val3>, ..." into a vector:
template <typename T>
std::vector<T> parse_list_values(std::string_view list_spec)
{
  std::vector<T> result;

  static const std::regex value_regex{
    "\\s*"     // Whitespace
    "([^,]+?)" // Single value
    "\\s*"     // Whitespace
    "(?:,|$)"  // Delimiters
  };

  auto values_begin =
    sv_regex_iterator(list_spec.cbegin(), list_spec.cend(), value_regex);
  auto values_end = sv_regex_iterator{};
  while (values_begin != values_end)
  {
    auto match          = *values_begin++;
    std::string_view sv = submatch_to_sv(match[1]);

    T val;
    parse(sv, val);
    result.push_back(std::move(val));
  }

  return result;
}

// Parses a range specification "<start> : <stop> [ : <stride> ]" and returns
// a vector filled with the specified range.
template <typename T>
std::vector<T> parse_range_values(std::string_view range_spec,
                                  nvbench::wrapped_type<T>)
{
  std::vector<T> range_params;

  static const std::regex value_regex{
    "\\s*"     // Whitespace
    "([^:]+?)" // Single value
    "\\s*"     // Whitespace
    "(?:$|:)"  // Delimiters
  };

  auto values_begin =
    sv_regex_iterator(range_spec.cbegin(), range_spec.cend(), value_regex);
  auto values_end = sv_regex_iterator{};
  for (; values_begin != values_end; ++values_begin)
  {
    auto match          = *values_begin;
    std::string_view sv = submatch_to_sv(match[1]);
    T val;
    parse(sv, val);
    range_params.push_back(std::move(val));
  }

  // Convert the parsed values into a range:
  if (range_params.size() != 2 && range_params.size() != 3)
  {
    throw std::runtime_error(fmt::format("{}:{}: Expected 2 or 3 values for "
                                         "range specification: {}",
                                         __FILE__,
                                         __LINE__,
                                         range_spec));
  }

  const T first  = range_params[0];
  const T last   = range_params[1];
  const T stride = range_params.size() == 3 ? range_params[2] : T{1};

  return nvbench::range(first, last, stride);
}

// Disable range parsing for string types
std::vector<std::string> parse_range_values(std::string_view range_spec,
                                            nvbench::wrapped_type<std::string>)
{
  throw std::runtime_error(fmt::format("{}:{}: Cannot use range syntax for "
                                       "string axis specification: `{}`.",
                                       __FILE__,
                                       __LINE__,
                                       range_spec));
}

template <typename T>
std::vector<T> parse_values(std::string_view value_spec)
{
  static const std::regex list_regex{"\\{"        // Literal {
                                     "\\s*"       // Whitespace
                                     "([^\\}]+?)" // list of values
                                     "\\s*"       // Whitespace
                                     "\\}"};      // Literal }

  static const std::regex range_regex{"\\("        // Literal (
                                      "\\s*"       // Whitespace
                                      "([^\\)]+?)" // range spec
                                      "\\s*"       // Whitespace
                                      "\\)"};      // Literal )
  sv_match match;
  if (std::regex_search(value_spec.cbegin(),
                        value_spec.cend(),
                        match,
                        list_regex))
  {
    return parse_list_values<T>(submatch_to_sv(match[1]));
  }
  else if (std::regex_search(value_spec.cbegin(),
                             value_spec.cend(),
                             match,
                             range_regex))
  {
    return parse_range_values(submatch_to_sv(match[1]),
                              nvbench::wrapped_type<T>{});
  }
  else
  {
    throw std::runtime_error(fmt::format("{}:{}: Invalid axis value spec: {}",
                                         __FILE__,
                                         __LINE__,
                                         value_spec));
  }
}

// Parse an axis specification into a 3-tuple of string_views containing the
// axis name, flags, and values.
auto parse_axis_key_flag_value_spec(const std::string &spec)
{
  static const std::regex spec_regex{
    "\\s*"        // Optional Whitespace
    "([^\\[:]+?)" // Axis name
    "\\s*"        // Optional Whitespace

    "(?:"        // Start optional non-capture group for tag
    "\\["        //  - Literal [
    "\\s*"       //  - Optional Whitespace
    "([^\\]]*?)" //  - Flag spec
    "\\s*"       //  - Optional Whitespace
    "\\]"        //  - Literal ]
    ")?"         // End optional tag group

    "\\s*"  // Optional Whitespace
    ":"     // Literal :
    "\\s*"  // Optional Whitespace
    "(.+?)" // Value spec
    "\\s*"  // Optional Whitespace
    "$"     // end
  };

  sv_match match;
  const std::string_view spec_sv = spec;
  if (!std::regex_search(spec_sv.cbegin(), spec_sv.cend(), match, spec_regex))
  {
    throw std::runtime_error(
      fmt::format("{}:{}: Bad format.", __FILE__, __LINE__));
  }

  // Extract the matches:
  const auto name = submatch_to_sv(match[1]);
  const auto flag = submatch_to_sv(match[2]);
  const auto vals = submatch_to_sv(match[3]);
  return std::tie(name, flag, vals);
}

} // namespace

namespace nvbench
{

void option_parser::parse(int argc, char const *const *argv)
{
  m_args.clear();
  m_args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i)
  {
    m_args.emplace_back(argv[i]);
  }

  parse_impl();
}

void option_parser::parse(std::vector<std::string> args)
{
  m_args = std::move(args);
  parse_impl();
}

void option_parser::parse_impl()
{
  auto cur_arg       = m_args.cbegin();
  const auto arg_end = m_args.cend();

  // The first arg may be the executable name:
  if (cur_arg != arg_end && !cur_arg->empty() && cur_arg->front() != '-')
  {
    cur_arg++;
  }

  auto check_params = [&cur_arg, &arg_end](std::size_t num_params) {
    const std::size_t rem_args = std::distance(cur_arg, arg_end) - 1;
    if (rem_args < num_params)
    {
      throw std::runtime_error(fmt::format("{}:{}: Option '{}' requires {} "
                                           "parameters, {} provided.",
                                           __FILE__,
                                           __LINE__,
                                           *cur_arg,
                                           num_params,
                                           rem_args));
    }
  };

  while (cur_arg < arg_end)
  {
    const auto &arg = *cur_arg;

    if (arg == "--benchmark" || arg == "-b")
    {
      check_params(1);
      this->add_benchmark(cur_arg[1]);
      cur_arg += 2;
    }
    else if (arg == "--axis" || arg == "-a")
    {
      check_params(1);
      this->update_axis(cur_arg[1]);
      cur_arg += 2;
    }
    else
    {
      throw std::runtime_error(fmt::format("{}:{}: Unrecognized command-line "
                                           "argument: `{}`.",
                                           __FILE__,
                                           __LINE__,
                                           arg));
    }
  }
}

void option_parser::add_benchmark(const std::string &name)
{
  const auto &mgr = nvbench::benchmark_manager::get();
  m_benchmarks.push_back(mgr.get_benchmark(name).clone());
}

void option_parser::update_axis(const std::string &spec)
{
  // Valid examples:
  // - "NumInputs [pow2] : (10 : 30 : 5)" <- Range specification (::)
  // - "UniqueKeys [] : { 10, 15, 20, 25, 30 }"  <- List spec {,,...}
  // - "Quality : (0.0 : 1.0 : 0.1)"
  // - "ValueType : { I32, F32, U64 }"
  // - "RNG [] : { Uniform, Gaussian }"
  //
  // Generally: "<AxisName> [<optional flags>] : <input spec>"
  //
  // Axis/Flag spec: "<AxisName>" (no flags)
  // Axis/Flag spec: "<AxisName> []" (no flags)
  // Axis/Flag spec: "<AxisName> [pow2]" (flags=`pow2`)
  // Value spec: "{ <v1, <v2>, ... }" <- Explicit values
  // Value spec: "(<start> : <stop>)" <- Range, inclusive start/stop
  // Value spec: "(<start> : <stop> : <stride>)" <- Range, explicit stride

  // Check that an active benchmark exists:
  if (m_benchmarks.empty())
  {
    throw std::runtime_error(fmt::format("{}:{}: \"--axis <...>\" must follow "
                                         "\"--benchmark <...>\".",
                                         __FILE__,
                                         __LINE__));
  }
  benchmark_base &bench = *m_benchmarks.back();

  try
  {
    const auto [name, flags, values] = parse_axis_key_flag_value_spec(spec);
    nvbench::axis_base &axis         = bench.get_axes().get_axis(name);
    switch (axis.get_type())
    {
      case axis_type::type:
        this->update_type_axis(static_cast<nvbench::type_axis &>(axis),
                               values,
                               flags);
        break;

      case axis_type::int64:
        this->update_int64_axis(static_cast<nvbench::int64_axis &>(axis),
                                values,
                                flags);
        break;

      case axis_type::float64:
        this->update_float64_axis(static_cast<nvbench::float64_axis &>(axis),
                                  values,
                                  flags);

        break;

      case axis_type::string:
        this->update_string_axis(static_cast<nvbench::string_axis &>(axis),
                                 values,
                                 flags);

        break;

      default:
        // Internal error, this should never happen:
        throw std::runtime_error(
          fmt::format("{}:{}: Internal error: invalid axis type enum '{}'",
                      __FILE__,
                      __LINE__,
                      static_cast<int>(axis.get_type())));
    }
  }
  catch (std::runtime_error &err)
  {
    throw std::runtime_error(fmt::format("{}:{}: Error parsing `--axis` "
                                         "specification `{}`.\n{}",
                                         __FILE__,
                                         __LINE__,
                                         spec,
                                         err.what()));
  }
}

void option_parser::update_int64_axis(int64_axis &axis,
                                      std::string_view value_spec,
                                      std::string_view flag_spec)
{
  // Validate flags:
  int64_axis_flags flags;
  if (flag_spec.empty())
  {
    flags = int64_axis_flags::none;
  }
  else if (flag_spec == "pow2")
  {
    flags = int64_axis_flags::power_of_two;
  }
  else
  {
    throw std::runtime_error(fmt::format("{}:{}: Invalid flag for int64 axis: "
                                         "`{}`",
                                         __FILE__,
                                         __LINE__,
                                         flag_spec));
  }

  auto input_values = parse_values<nvbench::int64_t>(value_spec);

  axis.set_inputs(std::move(input_values), flags);
}

void option_parser::update_float64_axis(float64_axis &axis,
                                        std::string_view value_spec,
                                        std::string_view flag_spec)
{
  // Validate flags:
  if (!flag_spec.empty())
  {
    throw std::runtime_error(fmt::format("{}:{}: Invalid flag for float64 "
                                         "axis: `{}`",
                                         __FILE__,
                                         __LINE__,
                                         flag_spec));
  }

  auto input_values = parse_values<nvbench::float64_t>(value_spec);

  axis.set_inputs(std::move(input_values));
}

void option_parser::update_string_axis(string_axis &axis,
                                       std::string_view value_spec,
                                       std::string_view flag_spec)
{
  // Validate flags:
  if (!flag_spec.empty())
  {
    throw std::runtime_error(fmt::format("{}:{}: Invalid flag for string "
                                         "axis: `{}`",
                                         __FILE__,
                                         __LINE__,
                                         flag_spec));
  }

  auto input_values = parse_values<std::string>(value_spec);

  axis.set_inputs(std::move(input_values));
}

void option_parser::update_type_axis(type_axis &axis,
                                     std::string_view value_spec,
                                     std::string_view flag_spec)
{
  // Validate flags:
  if (!flag_spec.empty())
  {
    throw std::runtime_error(fmt::format("{}:{}: Invalid flag for type axis: "
                                         "`{}`",
                                         __FILE__,
                                         __LINE__,
                                         flag_spec));
  }

  auto input_values = parse_values<std::string>(value_spec);

  axis.set_active_inputs(input_values);
}

} // namespace nvbench