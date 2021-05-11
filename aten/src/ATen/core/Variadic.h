#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ATen/core/List.h>
#include <c10/util/ArrayRef.h>

namespace at {

// This class allows you to write variadic functions which
// call a (possibly overloaded) function on each argument,
// in order.  This is most commonly used in autogenerated code,
// where it is convenient to have a function that can uniformly
// take arguments of different types.  If your arguments
// are homogenous consider using a std::initializer_list instead.
//
// For examples of this in use, see torch/csrc/utils/variadic.h
template <typename F>
struct IterArgs {
  template <typename... Args>
  inline F& apply() {
    return self();
  }

  // NB: Use perfect forwarding here, otherwise we'll make value
  // copies of all arguments!
  template <typename T, typename... Args>
  inline F& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }

  // Here are some handy overloads which provide sensible
  // defaults for container-like structures that one might
  // be interested in recursing into.  You can enable them
  // by adding:
  //
  //    using IterArgs<YourStructName>::operator()
  //
  // to your struct.  These are not enabled by default because
  // you may be able to process these structures more efficiently
  // than handling them one-by-one.

  template <typename T>
  void operator()(at::ArrayRef<T> args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  template <typename T>
  void operator()(const torch::List<T>& args) {
    for (const auto& arg : args) {
      self()(arg);
      if (self().short_circuit())
        return;
    }
  }

  // NB: we need to specify std::vector manually as C++ won't
  // do an implicit conversion to make a template deduction go through.
  template <typename T>
  void operator()(const std::vector<T>& args) {
    self()(at::ArrayRef<T>{args});
  }

  constexpr bool short_circuit() const {
    return false;
  }

 private:
  inline F& self() {
    return *static_cast<F*>(this);
  }
};

} // namespace at
