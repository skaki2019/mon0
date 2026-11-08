#include "crs_dataset.hpp"
#include "transition_matrix.hpp"
#include "multi_order_network.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/intrusive/counter.h>
#include <nanobind/intrusive/ref.h>

#include <nanobind/intrusive/counter.inl>

namespace nb = nanobind;

// ============================================================================
// nanobind bindings
// ============================================================================
NB_MODULE(_impl, m) {
  m.doc() = "CSR Dataset and Transition Matrix library";

  nb::intrusive_init(
      [](PyObject *o) noexcept {
#if !defined(NB_FREE_THREADED)
        Py_INCREF(o);
#else
        nb::gil_scoped_acquire guard;
        if (guard.is_valid())
          Py_INCREF(o);
#endif
      },
      [](PyObject *o) noexcept {
#if !defined(NB_FREE_THREADED)
        Py_DECREF(o);
#else
        nb::gil_scoped_acquire guard;
        if (guard.is_valid())
          Py_DECREF(o);
#endif
      });

// CSRDataset class
  nb::class_<CSRDataset>(
      m, "CSRDataset",
      nb::intrusive_ptr<CSRDataset>(
          [](CSRDataset *o, PyObject *po) noexcept { o->set_self_py(po); }))
      // Constructors
      .def(nb::init<const std::vector<std::vector<int>> &, int, int>(),
           nb::arg("edge_list"), nb::arg("n_nodes"), nb::arg("delta"),
           "Construct from list of [u, v, t] edges")
      // Accessors
      .def("num_nodes", &CSRDataset::num_nodes)
      .def("delta", &CSRDataset::delta)
      // Debug: return to Python
      .def("to_python_lists", &CSRDataset::to_python_lists,
           "Return (row_ptr, col_idx, edge_data) as Python lists")
      // Binary I/O
      .def("save_binary", &CSRDataset::save_binary, nb::arg("path"),
           "Save dataset to binary file")
      .def_static("load_binary", &CSRDataset::load_binary, nb::arg("path"),
                  "Load dataset from binary file");

  nb::class_<TransitionMatrix0>(
      m, "TransitionMatrix0",
      nb::intrusive_ptr<TransitionMatrix0>(
          [](TransitionMatrix0 *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def(nb::init<nb::ref<CSRDataset>>())
      .def("build_matrix", &TransitionMatrix0::build_matrix)
      .def_prop_ro("bayesian_sc",
                   [](const TransitionMatrix0 &self) { return self.bayesian_sc_; })
      .def_prop_ro("log_prob",
                   [](const TransitionMatrix0 &self) { return self.nodes_log_prob_; });

  // TransitionMatrix
  nb::class_<TransitionMatrix>(
      m, "TransitionMatrix",
      nb::intrusive_ptr<TransitionMatrix>(
          [](TransitionMatrix *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def(nb::init<nb::ref<CSRDataset>, const int>())
      .def("build_matrix", &TransitionMatrix::build_matrix)
      .def("compute_transition_probability",
           &TransitionMatrix::compute_transition_probability, nb::arg("ord"))
      .def("save_tmat_binary", &TransitionMatrix::save_tmat_binary, nb::arg("path"))
      .def_static("load_tmat_binary", &TransitionMatrix::load_tmat_binary,
                  nb::arg("path"));

  nb::class_<MultiOrderNetwork>(
      m, "MultiOrderNetwork",
      nb::intrusive_ptr<MultiOrderNetwork>(
          [](MultiOrderNetwork *o, PyObject *po) noexcept { o->set_self_py(po); }))
      .def(nb::init<nb::ref<CSRDataset>, int>())
      // Acessors
      .def("dataset", &MultiOrderNetwork::dataset)
      .def("max_order", &MultiOrderNetwork::max_order)
      .def("num_order", &MultiOrderNetwork::num_orders)
      .def("get_matrix", &MultiOrderNetwork::get_matrix)
      .def("build_matrix", &MultiOrderNetwork::build_matrix)
      .def("compute_probability", &MultiOrderNetwork::compute_probability)
      .def("compute_loglikelihood", &MultiOrderNetwork::compute_loglikelihood)
      .def("compute_dof", &MultiOrderNetwork::degrees_of_freedom)
      .def_prop_ro("transition_matrices",
                   [](const MultiOrderNetwork &self) { return self.matrices_; })
      .def_prop_ro("matrix0",
                   [](const MultiOrderNetwork &self) { return self.matrix0_; })
      .def_prop_ro("dofv", [](const MultiOrderNetwork &self) { return self.dof; })
      .def_prop_ro("dofaccv", [](const MultiOrderNetwork &self) { return self.dof_acc; });
}