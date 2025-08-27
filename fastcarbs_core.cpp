#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ========================= Precision Switch =========================
using Real = float;

using MatrixX    = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using RowMatrixX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorX    = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using RowVectorX = Eigen::Matrix<Real, 1, Eigen::Dynamic>;
using ArrayXX    = Eigen::Array<Real,  Eigen::Dynamic, Eigen::Dynamic>;

constexpr Real kPI = static_cast<Real>(3.14159265358979323846264338327950288L);

// ========================= NumPy <-> Eigen utils =========================
inline MatrixX as_eigen_2d(py::array_t<Real, py::array::c_style | py::array::forcecast> a) {
    if (a.ndim() != 2) throw std::invalid_argument("Expected a 2D array.");
    auto buf = a.request();
    auto* ptr = static_cast<Real*>(buf.ptr);
    // NumPy c_style gives us contiguous row-major memory
    Eigen::Map<const RowMatrixX> M(ptr, static_cast<Eigen::Index>(buf.shape[0]),
                                        static_cast<Eigen::Index>(buf.shape[1]));
    return MatrixX(M);
}

inline VectorX as_eigen_vec(py::array_t<Real, py::array::c_style | py::array::forcecast> a) {
    if (a.ndim() == 1) {
        auto buf = a.request();
        auto* ptr = static_cast<Real*>(buf.ptr);
        Eigen::Map<const VectorX> v(ptr, static_cast<Eigen::Index>(buf.shape[0]));
        return VectorX(v);
    } else if (a.ndim() == 2) {
        auto buf = a.request();
        auto* ptr = static_cast<Real*>(buf.ptr);
        if (buf.shape[1] == 1) {
            Eigen::Map<const VectorX> v(ptr, static_cast<Eigen::Index>(buf.shape[0]));
            return VectorX(v);
        } else if (buf.shape[0] == 1) {
            Eigen::Map<const RowVectorX> v(ptr, static_cast<Eigen::Index>(buf.shape[1]));
            return VectorX(v.transpose());
        } else {
            throw std::invalid_argument("y must be shape (N,), (N,1), or (1,N).");
        }
    } else {
        throw std::invalid_argument("y must be 1D or 2D.");
    }
}

inline py::array_t<Real> eigen_to_numpy(const VectorX& v) {
    py::array_t<Real> out(v.size());
    auto buf = out.request();
    auto* ptr = static_cast<Real*>(buf.ptr);
    Eigen::Map<VectorX>(ptr, v.size()) = v;
    return out;
}

// ========================= Kernel Base =========================
class Kernel {
public:
    virtual ~Kernel() = default;
    virtual std::string name() const = 0;
    virtual size_t input_dim() const = 0;

    virtual MatrixX K(const MatrixX& X, const MatrixX& Z) const = 0;
    virtual MatrixX K(const MatrixX& X) const { return K(X, X); }

    virtual VectorX Kdiag(const MatrixX& X) const {
        MatrixX Kxx = K(X, X);
        return Kxx.diagonal();
    }

    virtual size_t   num_params() const = 0;
    virtual void     get_unconstrained(VectorX& u, size_t& off) const = 0;
    virtual void     set_unconstrained(const VectorX& u, size_t& off) = 0;
    virtual void     add_grad_wrt_unconstrained(const MatrixX& X,
                                                const MatrixX& B,
                                                VectorX& grad,
                                                size_t& off) const = 0;
    virtual py::dict hypers_dict() const = 0;
};

// ========================= Matern 3/2 ARD =========================
// k(x,z) = sigma2 * (1 + sqrt(3) r) * exp(-sqrt(3) r)
// where r = sqrt( sum_j ((x_j - z_j)^2 / ell_j^2) )
class Matern32ARD final : public Kernel {
public:
    Matern32ARD(size_t input_dim, Real init_lengthscale, Real init_variance = Real(1))
        : D_(input_dim),
          log_ell_(VectorX::Constant(static_cast<Eigen::Index>(input_dim),
                                     std::log(init_lengthscale))),
          log_sigma2_(std::log(init_variance)) {}

    std::string name() const override { return "Matern32ARD"; }
    size_t input_dim() const override { return D_; }

    MatrixX K(const MatrixX& X, const MatrixX& Z) const override {
        if (static_cast<size_t>(X.cols()) != D_ || static_cast<size_t>(Z.cols()) != D_) {
            throw std::invalid_argument("Matern32ARD: Input dimension mismatch.");
        }
        const Real sigma2 = std::exp(log_sigma2_);
        const Real a = std::sqrt(Real(3));

        MatrixX q = MatrixX::Zero(X.rows(), Z.rows()); // sum_j ...
        for (size_t j = 0; j < D_; ++j) {
            const Real inv_l2 = std::exp(Real(-2) * log_ell_(static_cast<Eigen::Index>(j))); // 1/ell^2
            VectorX xj = X.col(static_cast<Eigen::Index>(j));
            VectorX zj = Z.col(static_cast<Eigen::Index>(j));
            VectorX xj2 = xj.array().square().matrix();
            VectorX zj2 = zj.array().square().matrix();
            MatrixX S = xj2.replicate(1, Z.rows())
                      + zj2.transpose().replicate(X.rows(), 1)
                      - Real(2) * (xj * zj.transpose());
            q.noalias() += inv_l2 * S;
        }
        MatrixX r = q.array().max(Real(0)).sqrt().matrix();
        ArrayXX E = (-a * r.array()).exp();               // exp(-a r)
        ArrayXX base = (Real(1) + a * r.array()) * E;     // (1 + a r) exp(-a r)
        MatrixX K = (sigma2 * base).matrix();
        return K;
    }

    size_t num_params() const override { return D_ + 1; }

    void get_unconstrained(VectorX& u, size_t& off) const override {
        for (size_t j = 0; j < D_; ++j) u(static_cast<Eigen::Index>(off++)) = log_ell_(static_cast<Eigen::Index>(j));
        u(static_cast<Eigen::Index>(off++)) = log_sigma2_;
    }

    void set_unconstrained(const VectorX& u, size_t& off) override {
        for (size_t j = 0; j < D_; ++j) log_ell_(static_cast<Eigen::Index>(j)) = u(static_cast<Eigen::Index>(off++));
        log_sigma2_ = u(static_cast<Eigen::Index>(off++));
    }

    void add_grad_wrt_unconstrained(const MatrixX& X,
                                    const MatrixX& B,
                                    VectorX& grad,
                                    size_t& off) const override {
        const Eigen::Index N = X.rows();
        const Real a = std::sqrt(Real(3));
        const Real sigma2 = std::exp(log_sigma2_);

        // Precompute S_j and q
        std::vector<MatrixX> Sj(D_);
        MatrixX q = MatrixX::Zero(N, N);
        for (size_t j = 0; j < D_; ++j) {
            VectorX xj  = X.col(static_cast<Eigen::Index>(j));
            VectorX xj2 = xj.array().square().matrix();
            MatrixX S = xj2.replicate(1, N) + xj2.transpose().replicate(N, 1)
                      - Real(2) * (xj * xj.transpose());
            Sj[j] = std::move(S);
        }
        for (size_t j = 0; j < D_; ++j) {
            const Real inv_l2 = std::exp(Real(-2) * log_ell_(static_cast<Eigen::Index>(j)));
            q.noalias() += inv_l2 * Sj[j];
        }
        MatrixX r = q.array().max(Real(0)).sqrt().matrix();
        ArrayXX E = (-a * r.array()).exp();
        ArrayXX base = (Real(1) + a * r.array()) * E;

        // dK/dsigma2 = base
        MatrixX dK_dsigma2 = base.matrix();
        Real g_sigma2 = Real(0.5) * (B.cwiseProduct(dK_dsigma2)).sum();
        grad(static_cast<Eigen::Index>(off + D_)) += g_sigma2 * std::exp(log_sigma2_); // chain to log_sigma2

        // dK/dell_j = sigma2 * a^2 * exp(-a r) * S_j / ell^3
        const Real a2 = a * a;
        for (size_t j = 0; j < D_; ++j) {
            const Real ell   = std::exp(log_ell_(static_cast<Eigen::Index>(j)));
            const Real inv_l3 = Real(1) / (ell * ell * ell);
            MatrixX dK_dell = (sigma2 * a2) * (E.matrix().cwiseProduct(Sj[j])) * inv_l3;
            Real g_ell = Real(0.5) * (B.cwiseProduct(dK_dell)).sum();
            grad(static_cast<Eigen::Index>(off + j)) += g_ell * ell; // chain to log_ell
        }

        off += D_ + 1;
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"] = py::str("Matern32ARD");
        py::array_t<Real> ls(D_);
        {
            auto buf = ls.request();
            auto* ptr = static_cast<Real*>(buf.ptr);
            for (size_t j = 0; j < D_; ++j)
                ptr[j] = static_cast<Real>(std::exp(log_ell_(static_cast<Eigen::Index>(j))));
        }
        d["lengthscales"] = ls;
        d["variance"]     = static_cast<double>(std::exp(log_sigma2_)); // cast to python float64
        return d;
    }

private:
    size_t  D_;
    VectorX log_ell_;   // log lengthscales (ARD)
    Real    log_sigma2_;
};

// ========================= Linear Kernel =========================
// k(x,z) = sigma2 * x . z
class LinearKernel final : public Kernel {
public:
    LinearKernel(size_t input_dim, Real init_variance = Real(1))
        : D_(input_dim), log_sigma2_(std::log(init_variance)) {}

    std::string name() const override { return "Linear"; }
    size_t input_dim() const override { return D_; }

    MatrixX K(const MatrixX& X, const MatrixX& Z) const override {
        if (static_cast<size_t>(X.cols()) != D_ || static_cast<size_t>(Z.cols()) != D_) {
            throw std::invalid_argument("LinearKernel: Input dimension mismatch.");
        }
        const Real sigma2 = std::exp(log_sigma2_);
        MatrixX K = sigma2 * (X * Z.transpose());
        return K;
    }

    size_t num_params() const override { return 1; }
    void get_unconstrained(VectorX& u, size_t& off) const override { u(static_cast<Eigen::Index>(off++)) = log_sigma2_; }
    void set_unconstrained(const VectorX& u, size_t& off) override { log_sigma2_ = u(static_cast<Eigen::Index>(off++)); }

    void add_grad_wrt_unconstrained(const MatrixX& X,
                                    const MatrixX& B,
                                    VectorX& grad,
                                    size_t& off) const override {
        MatrixX G = X * X.transpose(); // dK/dsigma2
        Real g_sigma2 = Real(0.5) * (B.cwiseProduct(G)).sum();
        grad(static_cast<Eigen::Index>(off)) += g_sigma2 * std::exp(log_sigma2_);
        off += 1;
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"]   = py::str("Linear");
        d["variance"] = static_cast<double>(std::exp(log_sigma2_));
        return d;
    }

private:
    size_t D_;
    Real   log_sigma2_;
};

// ========================= Sum Kernel =========================
class SumKernel final : public Kernel {
public:
    SumKernel(std::unique_ptr<Kernel> k1, std::unique_ptr<Kernel> k2)
        : k1_(std::move(k1)), k2_(std::move(k2)) {
        if (k1_->input_dim() != k2_->input_dim())
            throw std::invalid_argument("SumKernel: subkernels must have same input_dim.");
    }

    std::string name() const override { return "Sum"; }
    size_t input_dim() const override { return k1_->input_dim(); }

    MatrixX K(const MatrixX& X, const MatrixX& Z) const override {
        return k1_->K(X, Z) + k2_->K(X, Z);
    }

    size_t num_params() const override { return k1_->num_params() + k2_->num_params(); }

    void get_unconstrained(VectorX& u, size_t& off) const override {
        k1_->get_unconstrained(u, off);
        k2_->get_unconstrained(u, off);
    }

    void set_unconstrained(const VectorX& u, size_t& off) override {
        k1_->set_unconstrained(u, off);
        k2_->set_unconstrained(u, off);
    }

    void add_grad_wrt_unconstrained(const MatrixX& X,
                                    const MatrixX& B,
                                    VectorX& grad,
                                    size_t& off) const override {
        k1_->add_grad_wrt_unconstrained(X, B, grad, off);
        k2_->add_grad_wrt_unconstrained(X, B, grad, off);
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"] = py::str("Sum");
        d["k1"] = k1_->hypers_dict();
        d["k2"] = k2_->hypers_dict();
        return d;
    }

private:
    std::unique_ptr<Kernel> k1_, k2_;
};

// ========================= RBF 1D =========================
// k(x,z) = sigma2 * exp( -0.5 * (x - z)^2 / ell^2 ) ; 1D only
class RBF1D final : public Kernel {
public:
    RBF1D(Real init_lengthscale = Real(1), Real init_variance = Real(1))
        : log_ell_(std::log(init_lengthscale)), log_sigma2_(std::log(init_variance)) {}

    std::string name() const override { return "RBF1D"; }
    size_t input_dim() const override { return 1; }

    MatrixX K(const MatrixX& X, const MatrixX& Z) const override {
        if (X.cols() != 1 || Z.cols() != 1)
            throw std::invalid_argument("RBF1D expects inputs with shape (N,1).");
        const Real ell     = std::exp(log_ell_);
        const Real sigma2  = std::exp(log_sigma2_);
        const Real inv_l2  = Real(1) / (ell * ell);

        VectorX x  = X.col(0);
        VectorX z  = Z.col(0);
        VectorX x2 = x.array().square().matrix();
        VectorX z2 = z.array().square().matrix();
        MatrixX S = x2.replicate(1, Z.rows()) + z2.transpose().replicate(X.rows(), 1)
                  - Real(2) * (x * z.transpose());
        MatrixX K = (sigma2 * (-Real(0.5) * inv_l2 * S.array()).exp()).matrix();
        return K;
    }

    size_t num_params() const override { return 2; }
    void get_unconstrained(VectorX& u, size_t& off) const override { u(static_cast<Eigen::Index>(off++)) = log_ell_; u(static_cast<Eigen::Index>(off++)) = log_sigma2_; }
    void set_unconstrained(const VectorX& u, size_t& off) override { log_ell_ = u(static_cast<Eigen::Index>(off++)); log_sigma2_ = u(static_cast<Eigen::Index>(off++)); }

    void add_grad_wrt_unconstrained(const MatrixX& X,
                                    const MatrixX& B,
                                    VectorX& grad,
                                    size_t& off) const override {
        if (X.cols() != 1) throw std::invalid_argument("RBF1D expects inputs with shape (N,1).");
        const Eigen::Index N = X.rows();
        const Real ell    = std::exp(log_ell_);
        const Real sigma2 = std::exp(log_sigma2_);
        const Real inv_l2 = Real(1) / (ell * ell);

        VectorX x  = X.col(0);
        VectorX x2 = x.array().square().matrix();
        MatrixX S = x2.replicate(1, N) + x2.transpose().replicate(N, 1)
                  - Real(2) * (x * x.transpose());
        MatrixX base = (-Real(0.5) * inv_l2 * S.array()).exp().matrix(); // exp(-0.5 S/ell^2)

        // dK/dsigma2 = base
        MatrixX dK_dsigma2 = base;
        Real g_sigma2 = Real(0.5) * (B.cwiseProduct(dK_dsigma2)).sum();
        grad(static_cast<Eigen::Index>(off + 1)) += g_sigma2 * sigma2; // chain to log_sigma2

        // dK/dell = sigma2 * base * (S / ell^3)
        Real inv_l3 = Real(1) / (ell * ell * ell);
        MatrixX dK_dell = sigma2 * base.cwiseProduct(S) * inv_l3;
        Real g_ell = Real(0.5) * (B.cwiseProduct(dK_dell)).sum();
        grad(static_cast<Eigen::Index>(off + 0)) += g_ell * ell;       // chain to log_ell

        off += 2;
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"]      = py::str("RBF1D");
        d["lengthscale"] = static_cast<double>(std::exp(log_ell_));
        d["variance"]    = static_cast<double>(std::exp(log_sigma2_));
        return d;
    }

private:
    Real log_ell_;
    Real log_sigma2_;
};

// ========================= GP Regression =========================
class GPRegression {
public:
    GPRegression(std::unique_ptr<Kernel> kernel, Real noise_var = Real(1e-2), Real jitter = Real(1e-4))
        : kernel_(std::move(kernel)), log_noise_(std::log(noise_var)), jitter_(jitter) {
        if (!kernel_) throw std::invalid_argument("Kernel must not be null.");
    }

    void set_data(py::array_t<Real> X_in, py::array_t<Real> y_in) {
        X_ = as_eigen_2d(X_in);
        y_ = as_eigen_vec(y_in);
        const auto N = X_.rows();
        if (y_.rows() != N)
            throw std::invalid_argument("set_data: X and y have incompatible first dimension.");
        L_.resize(0,0);
    }

    // Adam on MAP objective: NLML + (-log prior(noise))
    void fit(int num_steps = 1000, Real lr = Real(1e-4)) {
        if (X_.size() == 0) throw std::runtime_error("No data set. Call set_data first.");
        const Eigen::Index N = X_.rows();

        // flatten params: [kernel params (unconstrained), log_noise]
        const size_t Pk = kernel_->num_params();
        const size_t P  = Pk + 1;
        VectorX u(static_cast<Eigen::Index>(P));
        {
            size_t off = 0;
            kernel_->get_unconstrained(u, off);
            u(static_cast<Eigen::Index>(off++)) = log_noise_;
        }

        // Adam state
        const Real beta1 = Real(0.9), beta2 = Real(0.999), eps = Real(1e-8);
        VectorX m = VectorX::Zero(static_cast<Eigen::Index>(P));
        VectorX v = VectorX::Zero(static_cast<Eigen::Index>(P));

        // LogNormal prior for noise
        const Real mu = std::log(Real(1e-2));
        const Real sigma = Real(0.5);
        const Real sigma2 = sigma * sigma;
        const Real halfNlog2pi = Real(0.5) * static_cast<Real>(N) * std::log(Real(2) * kPI);

        for (int t = 1; t <= num_steps; ++t) {
            // unpack
            size_t off = 0;
            kernel_->set_unconstrained(u, off);
            log_noise_ = u(static_cast<Eigen::Index>(off++));
            const Real noise = std::exp(log_noise_);

            // K = K(X,X) + (noise + jitter) I
            MatrixX K = kernel_->K(X_);
            K.diagonal().array() += (noise + jitter_);

            // Cholesky
            Eigen::LLT<MatrixX> llt(K);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("Cholesky factorization failed. Try larger jitter.");
            }
            MatrixX L = llt.matrixL();

            // alpha = K^{-1} y
            VectorX alpha = llt.solve(y_);

            // logdet K = 2 * sum(log diag(L))
            Real logdet = Real(0);
            for (Eigen::Index i = 0; i < L.rows(); ++i) logdet += std::log(L(i,i));
            logdet *= Real(2);

            // NLML
            Real nll = Real(0.5) * y_.dot(alpha) + Real(0.5) * logdet + halfNlog2pi;

            // Negative log prior for noise ~ LogNormal(mu, sigma)
            const Real logn = log_noise_;
            Real nlp_noise = logn + std::log(sigma * std::sqrt(Real(2)*kPI))
                             + (logn - mu) * (logn - mu) / (Real(2) * sigma2);

            Real loss = nll + nlp_noise;
            (void)loss; // (not returned)

            // -------- gradient ----------
            // B = alpha alpha^T - K^{-1}
            MatrixX Kinv = llt.solve(MatrixX::Identity(N, N));
            MatrixX B = alpha * alpha.transpose() - Kinv;

            VectorX g = VectorX::Zero(static_cast<Eigen::Index>(P));
            // kernel hypers grad (w.r.t log params)
            off = 0;
            kernel_->add_grad_wrt_unconstrained(X_, B, g, off);

            // noise grad (marginal likelihood term): 0.5 * tr(B * dK/dnoise) * dnoise/dlognoise
            Real dL_dnoise = Real(0.5) * B.trace(); // dK/dnoise = I
            g(static_cast<Eigen::Index>(off)) += dL_dnoise * noise;

            // add prior grad wrt log noise: 1 + (logn - mu)/sigma^2
            g(static_cast<Eigen::Index>(off)) += Real(1) + (logn - mu) / sigma2;

            // Adam update
            m = beta1 * m + (Real(1) - beta1) * g;
            v = beta2 * v + (Real(1) - beta2) * g.array().square().matrix();

            VectorX mhat = m / (Real(1) - std::pow(beta1, static_cast<Real>(t)));
            VectorX vhat = v / (Real(1) - std::pow(beta2, static_cast<Real>(t)));

            VectorX step = (lr * mhat.array() / (vhat.array().sqrt() + eps)).matrix();
            u -= step;
        }

        // write back final params
        size_t off2 = 0;
        kernel_->set_unconstrained(u, off2);
        log_noise_ = u(static_cast<Eigen::Index>(off2++));

        L_.resize(0,0);
    }

    // predict: returns (mean, var) at Xq (diagonal), noiseless or noisy.
    std::pair<py::array_t<Real>, py::array_t<Real>>
    predict(py::array_t<Real> Xq_in, bool noiseless = true) const {
        if (X_.size() == 0) throw std::runtime_error("No data set. Call set_data first.");

        MatrixX Xq = as_eigen_2d(Xq_in);
        if (Xq.cols() != X_.cols())
            throw std::invalid_argument("predict: Xq has wrong feature dimension.");

        const Real noise = std::exp(log_noise_);

        MatrixX K = kernel_->K(X_);
        K.diagonal().array() += (noise + jitter_);
        Eigen::LLT<MatrixX> llt(K);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("Cholesky factorization failed in predict().");

        MatrixX L = llt.matrixL();
        VectorX alpha = llt.solve(y_);

        MatrixX KfXq = kernel_->K(X_, Xq); // N x M
        MatrixX Kqq  = kernel_->K(Xq, Xq); // M x M

        VectorX mean = (KfXq.transpose() * alpha);

        MatrixX v = L.template triangularView<Eigen::Lower>().solve(KfXq);
        VectorX var = Kqq.diagonal() - (v.array().square().colwise().sum()).matrix().transpose();

        if (!noiseless) var.array() += noise;

        return {eigen_to_numpy(mean), eigen_to_numpy(var)};
    }

    // sample from independent marginals N(mean_i, var_i)
    py::array_t<Real> sample_marginals(py::array_t<Real> Xq_in, bool noiseless = true, int seed = 0) const {
        auto mv = predict(Xq_in, noiseless);
        VectorX mean = as_eigen_vec(mv.first);
        VectorX var  = as_eigen_vec(mv.second);

        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        std::normal_distribution<Real> N01(Real(0), Real(1));

        VectorX s(mean.size());
        for (Eigen::Index i = 0; i < mean.size(); ++i) {
            Real stdv = std::sqrt(std::max(Real(0), var(i)));
            s(i) = mean(i) + stdv * N01(rng);
        }
        return eigen_to_numpy(s);
    }

    py::dict get_hypers() const {
        py::dict d;
        d["noise"]  = static_cast<double>(std::exp(log_noise_));
        d["jitter"] = static_cast<double>(jitter_);
        d["kernel"] = kernel_->hypers_dict();
        return d;
    }

private:
    std::unique_ptr<Kernel> kernel_;
    MatrixX X_;
    VectorX y_;
    mutable MatrixX L_; // placeholder for potential caching
    Real    log_noise_;
    Real    jitter_;
};

// ========================= PYBIND11 module =========================
PYBIND11_MODULE(fastcarbs_core, m) {
    py::class_<GPRegression, py::smart_holder>(m, "GPRegression")
        .def(py::init<std::unique_ptr<Kernel>, Real, Real>(),
             py::arg("kernel"), py::arg("noise_var")=Real(1e-2), py::arg("jitter")=Real(1e-4))
        .def("set_data", &GPRegression::set_data)
        .def("fit", &GPRegression::fit, py::arg("num_steps")=1000, py::arg("lr")=Real(1e-4))
        .def("predict", &GPRegression::predict, py::arg("Xq"), py::arg("noiseless")=true)
        .def("sample_marginals", &GPRegression::sample_marginals,
             py::arg("Xq"), py::arg("noiseless")=true, py::arg("seed")=0)
        .def("get_hypers", &GPRegression::get_hypers);

    py::class_<Kernel, py::smart_holder>(m, "Kernel");

    py::class_<Matern32ARD, Kernel, py::smart_holder>(m, "Matern32ARD")
        .def(py::init<size_t, Real, Real>(),
             py::arg("input_dim"), py::arg("init_lengthscale"),
             py::arg("init_variance")=Real(1));

    py::class_<LinearKernel, Kernel, py::smart_holder>(m, "LinearKernel")
        .def(py::init<size_t, Real>(),
             py::arg("input_dim"), py::arg("init_variance")=Real(1));

    py::class_<SumKernel, Kernel, py::smart_holder>(m, "SumKernel")
        .def(py::init<std::unique_ptr<Kernel>, std::unique_ptr<Kernel>>(),
             py::arg("k1"), py::arg("k2"));

    py::class_<RBF1D, Kernel, py::smart_holder>(m, "RBF1D")
        .def(py::init<Real, Real>(),
             py::arg("init_lengthscale")=Real(1), py::arg("init_variance")=Real(1));
}
