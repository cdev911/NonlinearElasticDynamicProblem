/* ===================================
 * 2D Plane Stress Neo-Hookean Cantilever Benchmark
 * Dynamic Analysis with Half-Sine Pulse Loading
 * ===================================
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/elasticity/kinematics.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>

namespace DynamicElasticity
{
  using namespace dealii;

  // ========================================
  // CSV Logger
  // ========================================
  
  class CSVLogger {
  private:
    std::ofstream file;
    
  public:
    CSVLogger(const std::string& filename, 
              const std::vector<std::string>& columns)
    {
      file.open(filename);
      for (size_t i = 0; i < columns.size(); ++i) {
        file << columns[i];
        if (i < columns.size() - 1) file << ",";
      }
      file << "\n";
    }
    
    template<typename... Args>
    void write_row(Args... values) {
      write_impl(values...);
      file << "\n";
      file.flush();
    }
    
    ~CSVLogger() { if (file.is_open()) file.close(); }
    
  private:
    template<typename T>
    void write_impl(T value) {
      file << std::setprecision(12) << value;
    }
    
    template<typename T, typename... Args>
    void write_impl(T value, Args... args) {
      file << std::setprecision(12) << value << ",";
      write_impl(args...);
    }
  };

  // ========================================
  // Simulation Parameters
  // ========================================
  
  struct SimulationParameters
  {
    double lambda, mu, density;
    double beta_newmark, gamma_newmark;
    double time_step, total_time;
    double alpha_M, alpha_K;
    double load_peak_force, load_ramp_time, thickness;
    double beam_length, beam_height;
    unsigned int global_refinement, subdivisions_x, subdivisions_y;
    unsigned int max_newton_iterations;
    double linear_solver_tolerance;
    unsigned int linear_solver_max_iterations;
    std::string output_directory;
    unsigned int output_interval;
    bool save_csv_logs;
    
    SimulationParameters()
      : lambda(16.442953e6), mu(335570.0), density(1100.0)
      , beta_newmark(0.25), gamma_newmark(0.5)
      , time_step(0.01), total_time(4.0)
      , alpha_M(0.0), alpha_K(0.0)
      , load_peak_force(50.0), load_ramp_time(0.5), thickness(0.1)
      , beam_length(1.0), beam_height(0.1)
      , global_refinement(4), subdivisions_x(10), subdivisions_y(1)
      , max_newton_iterations(20)
      , linear_solver_tolerance(1e-12), linear_solver_max_iterations(2000)
      , output_directory("output"), output_interval(10), save_csv_logs(true)
    {}
    
    void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material Properties");
      {
        prm.declare_entry("lambda", "16442953.0", Patterns::Double(0));
        prm.declare_entry("mu", "335570.0", Patterns::Double(0));
        prm.declare_entry("density", "1100.0", Patterns::Double(0));
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Time Integration");
      {
        prm.declare_entry("beta", "0.25", Patterns::Double(0, 0.5));
        prm.declare_entry("gamma", "0.5", Patterns::Double(0, 1));
        prm.declare_entry("time step", "0.01", Patterns::Double(0));
        prm.declare_entry("total time", "4.0", Patterns::Double(0));
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Damping");
      {
        prm.declare_entry("alpha M", "0.0", Patterns::Double(0));
        prm.declare_entry("alpha K", "0.0", Patterns::Double(0));
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Loading");
      {
        prm.declare_entry("peak force", "50.0", Patterns::Double(0));
        prm.declare_entry("ramp time", "0.5", Patterns::Double(0));
        prm.declare_entry("thickness", "0.1", Patterns::Double(0));
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Mesh");
      {
        prm.declare_entry("length", "1.0", Patterns::Double(0));
        prm.declare_entry("height", "0.1", Patterns::Double(0));
        prm.declare_entry("global refinement", "4", Patterns::Integer(0));
        prm.declare_entry("subdivisions x", "10", Patterns::Integer(1));
        prm.declare_entry("subdivisions y", "1", Patterns::Integer(1));
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Solver");
      {
        prm.declare_entry("max newton iterations", "20", Patterns::Integer(1));
        prm.declare_entry("linear solver tolerance", "1.0e-12", Patterns::Double(0));
        prm.declare_entry("linear solver max iterations", "2000", Patterns::Integer(1));
      }
      prm.leave_subsection();
      
      prm.enter_subsection("Output");
      {
        prm.declare_entry("output directory", "output", Patterns::Anything());
        prm.declare_entry("output interval", "10", Patterns::Integer(1));
        prm.declare_entry("save csv logs", "true", Patterns::Bool());
      }
      prm.leave_subsection();
    }
    
    void parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material Properties");
      lambda = prm.get_double("lambda");
      mu = prm.get_double("mu");
      density = prm.get_double("density");
      prm.leave_subsection();
      
      prm.enter_subsection("Time Integration");
      beta_newmark = prm.get_double("beta");
      gamma_newmark = prm.get_double("gamma");
      time_step = prm.get_double("time step");
      total_time = prm.get_double("total time");
      prm.leave_subsection();
      
      prm.enter_subsection("Damping");
      alpha_M = prm.get_double("alpha M");
      alpha_K = prm.get_double("alpha K");
      prm.leave_subsection();
      
      prm.enter_subsection("Loading");
      load_peak_force = prm.get_double("peak force");
      load_ramp_time = prm.get_double("ramp time");
      thickness = prm.get_double("thickness");
      prm.leave_subsection();
      
      prm.enter_subsection("Mesh");
      beam_length = prm.get_double("length");
      beam_height = prm.get_double("height");
      global_refinement = prm.get_integer("global refinement");
      subdivisions_x = prm.get_integer("subdivisions x");
      subdivisions_y = prm.get_integer("subdivisions y");
      prm.leave_subsection();
      
      prm.enter_subsection("Solver");
      max_newton_iterations = prm.get_integer("max newton iterations");
      linear_solver_tolerance = prm.get_double("linear solver tolerance");
      linear_solver_max_iterations = prm.get_integer("linear solver max iterations");
      prm.leave_subsection();
      
      prm.enter_subsection("Output");
      output_directory = prm.get("output directory");
      output_interval = prm.get_integer("output interval");
      save_csv_logs = prm.get_bool("save csv logs");
      prm.leave_subsection();
    }
    
    void print() const
    {
      std::cout << "\n=== 2D Plane Stress Cantilever Benchmark ===\n";
      std::cout << "Material: λ=" << lambda << " Pa, μ=" << mu << " Pa, ρ=" << density << " kg/m³\n";
      std::cout << "Time: dt=" << time_step << "s, T=" << total_time 
                << "s (" << static_cast<int>(total_time/time_step) << " steps)\n";
      std::cout << "Damping: α_M=" << alpha_M << ", α_K=" << alpha_K << "\n";
      std::cout << "Loading: F_peak=" << load_peak_force << "N, t_ramp=" << load_ramp_time 
                << "s, thickness=" << thickness << "m\n";
      std::cout << "  Line load: f_y = " << -load_peak_force/thickness << " N/m (peak)\n";
      std::cout << "Beam: L=" << beam_length << "m, H=" << beam_height 
                << "m, ref=" << global_refinement << "\n";
      std::cout << "Output: " << output_directory << ", interval=" << output_interval << "\n";
      std::cout << "==========================================\n\n";
    }
  };

  // ========================================
  // Output Manager with PVD
  // ========================================
  
  template <int dim>
  class OutputManager
  {
  private:
    std::string output_dir;
    std::string vtu_dir;
    std::ofstream pvd_file;
    std::vector<std::pair<double, std::string>> time_and_files;
    
  public:
    OutputManager(const std::string& directory) : output_dir(directory)
    {
      mkdir(output_dir.c_str(), 0755);
      
      vtu_dir = output_dir + "/VTUs";
      mkdir(vtu_dir.c_str(), 0755);
      
      std::string pvd_filename = output_dir + "/solution.pvd";
      pvd_file.open(pvd_filename);
      
      if (!pvd_file.is_open()) {
        std::cerr << "ERROR: Could not open PVD file: " << pvd_filename << "\n";
        return;
      }
      
      pvd_file << "<?xml version=\"1.0\"?>\n";
      pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
      pvd_file << "  <Collection>\n";
      pvd_file.flush();  // Flush header immediately
      
      std::cout << "Output directory: " << output_dir << "\n";
      std::cout << "VTU directory: " << vtu_dir << "\n";
      std::cout << "PVD file: " << pvd_filename << "\n";
    }
    
    void write_vtu(const DoFHandler<dim> &dof_handler,
                   const Vector<double> &solution,
                   const Vector<double> &velocity,
                   const Vector<double> &acceleration,
                   const Vector<double> &external_force,
                   const double time,
                   const unsigned int timestep)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      
      std::vector<std::string> sol_names(dim, "displacement");
      std::vector<std::string> vel_names(dim, "velocity");
      std::vector<std::string> acc_names(dim, "acceleration");
      std::vector<std::string> force_names(dim, "external_force");
      
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vec_interp(dim, DataComponentInterpretation::component_is_part_of_vector);
      
      data_out.add_data_vector(solution, sol_names, DataOut<dim>::type_dof_data, vec_interp);
      data_out.add_data_vector(velocity, vel_names, DataOut<dim>::type_dof_data, vec_interp);
      data_out.add_data_vector(acceleration, acc_names, DataOut<dim>::type_dof_data, vec_interp);
      data_out.add_data_vector(external_force, force_names, DataOut<dim>::type_dof_data, vec_interp);
      
      Vector<double> time_field(dof_handler.n_dofs());
      time_field = time;
      data_out.add_data_vector(time_field, "time");
      
      data_out.build_patches();
      
      std::ostringstream full_filename;
      full_filename << vtu_dir << "/solution-" 
               << std::setw(6) << std::setfill('0') << timestep << ".vtu";
      
      std::ofstream output(full_filename.str());
      data_out.write_vtu(output);
      
      std::ostringstream relative_filename;
      relative_filename << "VTUs/solution-" 
                       << std::setw(6) << std::setfill('0') << timestep << ".vtu";
      
      time_and_files.push_back(std::make_pair(time, relative_filename.str()));
      
      // Update PVD file immediately
      update_pvd_file();
    }
    
  private:
    void update_pvd_file()
    {
      // Rewrite the entire PVD file with all entries so far
      pvd_file.close();
      pvd_file.open(output_dir + "/solution.pvd");
      
      pvd_file << "<?xml version=\"1.0\"?>\n";
      pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
      pvd_file << "  <Collection>\n";
      
      for (const auto &entry : time_and_files) {
        pvd_file << "    <DataSet timestep=\"" << entry.first 
                 << "\" file=\"" << entry.second << "\"/>\n";
      }
      
      pvd_file << "  </Collection>\n";
      pvd_file << "</VTKFile>\n";
      pvd_file.flush();
    }
    
  public:
    
    ~OutputManager()
    {
      // PVD file is already complete from update_pvd_file() calls
      // Just close it
      if (pvd_file.is_open()) {
        pvd_file.close();
      }
      
      std::cout << "\nTime-series complete: " << time_and_files.size() << " files\n";
      std::cout << "Open " << output_dir << "/solution.pvd in ParaView\n";
    }
  };

  // ========================================
  // Main Solver Class
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  class DynamicSolver
  {
  public:
    DynamicSolver(const std::string &parameter_file);
    void run();

  private:
    void setup_system();
    void setup_constraints();
    void setup_dynamics();
    void assemble_mass_matrix();
    void assemble_system();
    void assemble_effective_system(double time);
    
    void predict_newmark();
    void correct_newmark(const Vector<double>& delta_u);
    void solve_time_step();
    void solve_linear(Vector<double> &du);
    
    Vector<double> compute_external_force(double time) const;
    double compute_load_magnitude(double time) const;
    
    double compute_kinetic_energy() const;
    double compute_potential_energy() const;
    void update_work_and_dissipation();
    
    void initialize_loggers();
    void log_time_step(double solve_time);
    void log_newton(unsigned int iter, double res, double upd);
    
    SimulationParameters params;
    
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    FESystem<dim> fe;
    
    AffineConstraints<double> constraints;
    
    SparsityPattern sparsity;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> damping_matrix;
    SparseMatrix<double> effective_stiffness;
    
    Vector<double> solution;
    Vector<double> solution_delta;
    Vector<double> newton_update;
    Vector<double> residual;
    Vector<double> external_force;
    
    Vector<double> velocity, acceleration;
    Vector<double> velocity_pred, acceleration_pred, displacement_pred;
    
    double current_time;
    unsigned int time_step_number;
    unsigned int newton_iteration;
    
    double cumulative_work, cumulative_dissipation;
    double prev_work_rate, prev_diss_rate;
    
    types::global_dof_index monitor_dof_x, monitor_dof_y;
    
    std::unique_ptr<OutputManager<dim>> output_mgr;
    std::unique_ptr<CSVLogger> time_log;
    std::unique_ptr<CSVLogger> energy_log;
    std::unique_ptr<CSVLogger> newton_log;
    
    mutable TimerOutput timer;
  };

  // ========================================
  // Constructor
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  DynamicSolver<dim, ADTypeCode>::DynamicSolver(const std::string &parameter_file)
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , current_time(0.0)
    , time_step_number(0)
    , newton_iteration(0)
    , cumulative_work(0.0)
    , cumulative_dissipation(0.0)
    , prev_work_rate(0.0)
    , prev_diss_rate(0.0)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {
    ParameterHandler prm;
    params.declare_parameters(prm);
    
    try {
      prm.parse_input(parameter_file);
      params.parse_parameters(prm);
      std::cout << "Loaded: " << parameter_file << "\n";
    }
    catch (...) {
      std::cout << "Using default parameters\n";
    }
    
    params.print();
  }

  // ========================================
  // Setup
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    std::cout << "DOFs: " << dof_handler.n_dofs() << "\n";
    
    solution.reinit(dof_handler.n_dofs());
    solution_delta.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    residual.reinit(dof_handler.n_dofs());
    external_force.reinit(dof_handler.n_dofs());
    
    velocity.reinit(dof_handler.n_dofs());
    acceleration.reinit(dof_handler.n_dofs());
    velocity_pred.reinit(dof_handler.n_dofs());
    acceleration_pred.reinit(dof_handler.n_dofs());
    displacement_pred.reinit(dof_handler.n_dofs());
    
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity.copy_from(dsp);
    
    system_matrix.reinit(sparsity);
    mass_matrix.reinit(sparsity);
    damping_matrix.reinit(sparsity);
    effective_stiffness.reinit(sparsity);
    
    // Find monitor point (bottom-right corner): (L, -H/2)
    const double target_x = params.beam_length;
    const double target_y = -params.beam_height / 2.0;
    double min_dist = 1e10;
    
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        double dx = cell->vertex(v)[0] - target_x;
        double dy = cell->vertex(v)[1] - target_y;
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < min_dist) {
          min_dist = dist;
          monitor_dof_x = cell->vertex_dof_index(v, 0);
          monitor_dof_y = cell->vertex_dof_index(v, 1);
        }
      }
    }
    std::cout << "Monitor point (bottom-right): DOFs " << monitor_dof_x << ", " << monitor_dof_y << "\n";
    std::cout << "Target: (" << target_x << ", " << target_y << "), distance: " << min_dist << "\n";
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::setup_constraints()
  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    
    // Fixed at left edge (x=0, boundary_id = 0 in deal.II)
    VectorTools::interpolate_boundary_values(dof_handler, 0,
                                            Functions::ZeroFunction<dim>(dim),
                                            constraints);
    constraints.close();
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::setup_dynamics()
  {
    std::cout << "Assembling mass matrix...\n";
    assemble_mass_matrix();
    
    damping_matrix = 0.0;
    damping_matrix.add(params.alpha_M, mass_matrix);
    
    // Verify mass
    Vector<double> ones(dof_handler.n_dofs());
    Vector<double> M_ones(dof_handler.n_dofs());
    ones = 1.0;
    for (auto &c : constraints.get_lines())
      ones[c.index] = 0.0;
    
    mass_matrix.vmult(M_ones, ones);
    double total_mass = ones * M_ones;
    double expected = params.density * params.beam_length * params.beam_height;
    std::cout << "Mass check: computed=" << total_mass << ", expected=" << expected 
              << ", error=" << std::abs(total_mass-expected)/expected*100 << "%\n";
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::assemble_mass_matrix()
  {
    mass_matrix = 0;
    QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> fe_values(fe, quad, update_values | update_JxW_values);
    
    FullMatrix<double> cell_mass(fe.dofs_per_cell);
    std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
    
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      cell_mass = 0;
      fe_values.reinit(cell);
      
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
          unsigned int ci = fe.system_to_component_index(i).first;
          for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
            unsigned int cj = fe.system_to_component_index(j).first;
            if (ci == cj)
              cell_mass(i, j) += params.density * 
                                 fe_values.shape_value(i, q) *
                                 fe_values.shape_value(j, q) *
                                 fe_values.JxW(q);
          }
        }
      
      cell->get_dof_indices(dofs);
      constraints.distribute_local_to_global(cell_mass, dofs, mass_matrix);
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::assemble_system()
  {
    using ADHelper = Differentiation::AD::EnergyFunctional<ADTypeCode, double>;
    using ADNumber = typename ADHelper::ad_type;
    
    Vector<double> current_sol = solution;
    current_sol += solution_delta;
    
    QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> fe_values(fe, quad, update_values | update_gradients | update_JxW_values);
    
    FullMatrix<double> cell_matrix(fe.dofs_per_cell);
    Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
    
    const FEValuesExtractors::Vector u(0);
    
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);
      cell->get_dof_indices(dofs);
      
      ADHelper ad_helper(fe.dofs_per_cell);
      ad_helper.register_dof_values(current_sol, dofs);
      const std::vector<ADNumber> &dof_vals = ad_helper.get_sensitive_dof_values();
      
      std::vector<Tensor<2, dim, ADNumber>> grad_u(quad.size());
      fe_values[u].get_function_gradients_from_local_dof_values(dof_vals, grad_u);
      
      ADNumber energy = ADNumber(0.0);
      for (unsigned int q = 0; q < quad.size(); ++q) {
        Tensor<2, dim, ADNumber> F = Physics::Elasticity::Kinematics::F(grad_u[q]);
        SymmetricTensor<2, dim, ADNumber> C = Physics::Elasticity::Kinematics::C(F);
        ADNumber J = determinant(F);
        
        ADNumber psi = params.mu * 0.5 * (trace(C) - static_cast<double>(dim))
                     - params.mu * std::log(J)
                     + params.lambda * 0.5 * std::pow(std::log(J), 2.0);
        energy += psi * fe_values.JxW(q);
      }
      
      ad_helper.register_energy_functional(energy);
      ad_helper.compute_residual(cell_rhs);
      cell_rhs *= -1.0;
      ad_helper.compute_linearization(cell_matrix);
      
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, dofs, 
                                             system_matrix, residual);
    }
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::assemble_effective_system(double time)
  {
    system_matrix = 0.0;
    residual = 0.0;
    assemble_system();
    
    damping_matrix = 0.0;
    damping_matrix.add(params.alpha_M, mass_matrix);
    damping_matrix.add(params.alpha_K, system_matrix);
    
    external_force = compute_external_force(time);
    
    const double dt = params.time_step;
    const double beta = params.beta_newmark;
    const double gamma = params.gamma_newmark;
    
    effective_stiffness.copy_from(system_matrix);
    effective_stiffness.add(gamma/(beta*dt), damping_matrix);
    effective_stiffness.add(1.0/(beta*dt*dt), mass_matrix);
    
    Vector<double> damp_force(dof_handler.n_dofs());
    Vector<double> inert_force(dof_handler.n_dofs());
    damping_matrix.vmult(damp_force, velocity_pred);
    mass_matrix.vmult(inert_force, acceleration_pred);
    
    residual += external_force;
    residual -= damp_force;
    residual -= inert_force;
    
    constraints.condense(effective_stiffness);
    constraints.condense(residual);
  }

  // ========================================
  // Time Integration
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::predict_newmark()
  {
    const double dt = params.time_step;
    const double beta = params.beta_newmark;
    const double gamma = params.gamma_newmark;
    
    displacement_pred = solution;
    displacement_pred.add(dt, velocity);
    displacement_pred.add(dt*dt*(0.5-beta), acceleration);
    
    velocity_pred = velocity;
    velocity_pred.add(dt*(1.0-gamma), acceleration);
    
    acceleration_pred = 0.0;
    solution_delta = displacement_pred;
    solution_delta -= solution;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::correct_newmark(const Vector<double>& delta_u)
  {
    const double dt = params.time_step;
    const double beta = params.beta_newmark;
    const double gamma = params.gamma_newmark;
    
    acceleration_pred.add(1.0/(beta*dt*dt), delta_u);
    velocity_pred.add(dt*gamma/(beta*dt*dt), delta_u);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::solve_time_step()
  {
    Timer timer;
    timer.start();
    
    double abs_tol = 1e-10;
    double rel_tol = 1e-8;
    
    predict_newmark();
    
    double res0 = 0.0;
    for (newton_iteration = 0; newton_iteration < params.max_newton_iterations; ++newton_iteration) {
      // current_time has already been incremented in run(), so use it directly
      assemble_effective_system(current_time);
      
      double res = residual.l2_norm();
      if (newton_iteration == 0 && res > 1e-14) res0 = res;
      
      log_newton(newton_iteration, res, 0.0);
      
      if (newton_iteration > 0) {
        if (res < abs_tol || (res0 > 0 && res/res0 < rel_tol))
          break;
      }
      
      solve_linear(newton_update);
      solution_delta += newton_update;
      correct_newmark(newton_update);
    }
    
    timer.stop();
    
    solution += solution_delta;
    velocity = velocity_pred;
    acceleration = acceleration_pred;
    
    update_work_and_dissipation();
    log_time_step(timer.wall_time());
    
    solution_delta = 0.0;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::solve_linear(Vector<double> &du)
  {
    du = 0.0;
    SolverControl control(params.linear_solver_max_iterations, 
                         params.linear_solver_tolerance);
    SolverCG<Vector<double>> cg(control);
    PreconditionSSOR<SparseMatrix<double>> precond;
    precond.initialize(effective_stiffness, 1.2);
    cg.solve(effective_stiffness, du, residual, precond);
    constraints.distribute(du);
  }

  // ========================================
  // Loading: Half-Sine Pulse
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  double DynamicSolver<dim, ADTypeCode>::compute_load_magnitude(double time) const
  {
    // Half-sine pulse: Fy(t) = -Fmax * sin(pi*t/tramp) for 0 <= t <= tramp
    // Line load: fy = Fy/W (force per unit thickness)
    if (time <= params.load_ramp_time) {
      double Fy = -params.load_peak_force * std::sin(M_PI * time / params.load_ramp_time);
      return Fy / params.thickness;  // Line load in N/m
    }
    return 0.0;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  Vector<double> DynamicSolver<dim, ADTypeCode>::compute_external_force(double time) const
  {
    Vector<double> force(dof_handler.n_dofs());
    force = 0.0;
    
    const QGauss<dim-1> face_quad(fe.degree * 2 + 1);
    FEFaceValues<dim> fe_face(fe, face_quad, update_values | update_JxW_values);
    
    Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<unsigned int> dofs(fe.dofs_per_cell);
    
    double line_load = compute_load_magnitude(time);  // N/m (negative y-direction)
    
    // Apply load on right edge (x=L, boundary_id = 1)
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      if (!cell->at_boundary()) continue;
      cell_rhs = 0;
      
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        if (cell->face(f)->boundary_id() == 1) {  // Right edge
          fe_face.reinit(cell, f);
          for (unsigned int q = 0; q < face_quad.size(); ++q)
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
              int comp = fe.system_to_component_index(i).first;
              double Ni = fe_face.shape_value(i, q);
              if (comp == 1)  // y-component only
                cell_rhs(i) += Ni * line_load * fe_face.JxW(q);
            }
        }
      }
      cell->get_dof_indices(dofs);
      constraints.distribute_local_to_global(cell_rhs, dofs, force);
    }
    return force;
  }

  // ========================================
  // Energy
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  double DynamicSolver<dim, ADTypeCode>::compute_kinetic_energy() const
  {
    Vector<double> Mv(dof_handler.n_dofs());
    mass_matrix.vmult(Mv, velocity);
    return 0.5 * (velocity * Mv);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  double DynamicSolver<dim, ADTypeCode>::compute_potential_energy() const
  {
    QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> fe_values(fe, quad, update_gradients | update_JxW_values);
    const FEValuesExtractors::Vector u(0);
    
    double PE = 0.0;
    for (const auto &cell : dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      std::vector<Tensor<2, dim>> grad_u(quad.size());
      fe_values[u].get_function_gradients(solution, grad_u);
      
      for (unsigned int q = 0; q < quad.size(); ++q) {
        Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(grad_u[q]);
        SymmetricTensor<2, dim> C = Physics::Elasticity::Kinematics::C(F);
        double J = determinant(F);
        
        if (J > 0) {
          double psi = params.mu * 0.5 * (trace(C) - dim)
                     - params.mu * std::log(J)
                     + params.lambda * 0.5 * std::pow(std::log(J), 2.0);
          PE += psi * fe_values.JxW(q);
        }
      }
    }
    return PE;
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::update_work_and_dissipation()
  {
    double work_rate = external_force * velocity;
    Vector<double> Cv(dof_handler.n_dofs());
    damping_matrix.vmult(Cv, velocity);
    double diss_rate = velocity * Cv;
    
    const double dt = params.time_step;
    cumulative_work += 0.5 * dt * (work_rate + prev_work_rate);
    cumulative_dissipation += 0.5 * dt * (diss_rate + prev_diss_rate);
    
    prev_work_rate = work_rate;
    prev_diss_rate = diss_rate;
  }

  // ========================================
  // Logging
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::initialize_loggers()
  {
    if (!params.save_csv_logs) return;
    
    mkdir(params.output_directory.c_str(), 0755);
    
    time_log = std::make_unique<CSVLogger>(
      params.output_directory + "/time_history.csv",
      std::vector<std::string>{"time","disp_x","disp_y","vel_x","vel_y",
        "acc_x","acc_y","KE","PE","E_total","W_ext","D_visc",
        "load_line","newton_iters","residual","solve_time"});
    
    energy_log = std::make_unique<CSVLogger>(
      params.output_directory + "/energy_balance.csv",
      std::vector<std::string>{"time","KE","PE","E_total","W_ext","D_visc","error","error_pct"});
    
    newton_log = std::make_unique<CSVLogger>(
      params.output_directory + "/newton_convergence.csv",
      std::vector<std::string>{"step","iter","residual","normalized","update"});
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::log_time_step(double solve_time)
  {
    if (!params.save_csv_logs) return;
    
    double KE = compute_kinetic_energy();
    double PE = compute_potential_energy();
    double err = KE + PE - cumulative_work + cumulative_dissipation;
    double err_pct = cumulative_work > 1e-12 ? std::abs(err)/cumulative_work*100 : 0;
    
    time_log->write_row(current_time, solution[monitor_dof_x], solution[monitor_dof_y],
      velocity[monitor_dof_x], velocity[monitor_dof_y], 
      acceleration[monitor_dof_x], acceleration[monitor_dof_y],
      KE, PE, KE+PE, cumulative_work, cumulative_dissipation,
      compute_load_magnitude(current_time),
      newton_iteration, residual.l2_norm(), solve_time);
    
    energy_log->write_row(current_time, KE, PE, KE+PE, cumulative_work, 
                         cumulative_dissipation, err, err_pct);
  }

  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::log_newton(unsigned int iter, double res, double upd)
  {
    if (params.save_csv_logs)
      newton_log->write_row(time_step_number, iter, res, 0.0, upd);
  }

  // ========================================
  // Main Run
  // ========================================
  
  template <int dim, Differentiation::AD::NumberTypes ADTypeCode>
  void DynamicSolver<dim, ADTypeCode>::run()
  {
    std::cout << "\n=== Creating Mesh ===\n";
    Point<dim> p1, p2;
    // Beam centered at y=0: from y=-H/2 to y=+H/2
    p1(0) = 0.0;
    p1(1) = -params.beam_height / 2.0;
    p2(0) = params.beam_length;
    p2(1) = params.beam_height / 2.0;
    
    std::vector<unsigned int> reps = {params.subdivisions_x, params.subdivisions_y};
    GridGenerator::subdivided_hyper_rectangle(triangulation, reps, p1, p2, true);
    triangulation.refine_global(params.global_refinement);
    std::cout << "Cells: " << triangulation.n_active_cells() << "\n";
    std::cout << "Beam domain: [" << p1 << "] to [" << p2 << "]\n";
    
    setup_system();
    setup_constraints();
    setup_dynamics();
    
    output_mgr = std::make_unique<OutputManager<dim>>(params.output_directory);
    initialize_loggers();
    
    // Compute initial external force at t=0
    external_force = compute_external_force(0.0);
    
    output_mgr->write_vtu(dof_handler, solution, velocity, acceleration, 
                         external_force, 0.0, 0);
    
    std::cout << "\n=== Time Integration ===\n";
    unsigned int n_steps = static_cast<unsigned int>(params.total_time / params.time_step);
    std::cout << "Steps: " << n_steps << "\n\n";
    
    std::cout << std::setw(6) << "Step" << " | " << std::setw(8) << "Time" << " | "
              << std::setw(4) << "NR" << " | " << std::setw(12) << "Disp_y(P)" << " | "
              << std::setw(10) << "KE" << " | " << std::setw(10) << "PE" << "\n";
    std::cout << std::string(75, '-') << "\n";
    
    for (time_step_number = 1; time_step_number <= n_steps; ++time_step_number) {
      current_time += params.time_step;
      
      solve_time_step();
      
      double disp_y = solution[monitor_dof_y];
      double KE = compute_kinetic_energy();
      double PE = compute_potential_energy();
      
      if (time_step_number % 10 == 0 || time_step_number < 60) {
        std::cout << std::setw(6) << time_step_number << " | "
                  << std::setw(8) << std::fixed << std::setprecision(3) << current_time << " | "
                  << std::setw(4) << newton_iteration << " | "
                  << std::setw(12) << std::scientific << std::setprecision(4) << disp_y << " | "
                  << std::setw(10) << std::setprecision(2) << KE << " | " 
                  << std::setw(10) << PE << "\n";
      }
      
      if (time_step_number % params.output_interval == 0)
        output_mgr->write_vtu(dof_handler, solution, velocity, acceleration,
                             external_force, current_time, time_step_number);
    }
    
    output_mgr->write_vtu(dof_handler, solution, velocity, acceleration,
                         external_force, current_time, time_step_number);
    
    double peak_disp = solution[monitor_dof_y];
    std::cout << "\n=== Results ===\n";
    std::cout << "Final vertical displacement at monitor point: " << peak_disp << " m\n";
    std::cout << "Expected range: -0.40 to -0.50 m\n";
    
    std::cout << "\n=== Complete ===\n";
    timer.print_summary();
  }

} // namespace DynamicElasticity

// ========================================
// Main
// ========================================

int main(int argc, char *argv[])
{
  try {
    using namespace dealii;
    
    std::string param_file = (argc > 1) ? argv[1] : "benchmark_parameters.prm";
    
    std::cout << "\n=== 2D Plane Stress Neo-Hookean Cantilever Benchmark ===\n";
    std::cout << "Parameter file: " << param_file << "\n";
    
    constexpr Differentiation::AD::NumberTypes ADType =
        Differentiation::AD::NumberTypes::sacado_dfad_dfad;
    
    DynamicElasticity::DynamicSolver<2, ADType> solver(param_file);
    solver.run();
    
    return 0;
  }
  catch (std::exception &exc) {
    std::cerr << "\nException: " << exc.what() << "\n";
    return 1;
  }
}