/*
 * arrt (another random ray tracer)
 * Gavin Ridley
 * MIT 22.212 F2019
 *
 * NOTES: -> add quasi-monte carlo and compare convergence
 *        -> Add ability to plot lines across geometry plot
 *        -> Check time per integration
 *        -> Could use the exception system better
 */
#include<algorithm>
#include<array>
#include<cstdlib>
#include<experimental/filesystem>
#include<fstream>
#include<functional>
#include<iostream> 
#include<iomanip>
#include<map>
#include<string>
#include<sstream>
#include<utility>
#include<vector>

#include<math.h>
#include<gd.h>

#define EPS 1e-6f

// Know npolar at compile time to speed stuff up in core MOC kernel
// This speedup is due to loop unrolling.
#define NPOLAR 1

using namespace std;
namespace fs = experimental::filesystem;

// Crappy RNG
constexpr float randmaxf = (float)RAND_MAX;
float randf() { return (float)rand() / randmaxf; }

// useful operators on 2D points
typedef pair<float, float> Pt2D;
template<class T>
pair<T, T> operator+(pair<T,T> x, pair<T,T> y)
{
  pair<T, T> res;
  res.first = x.first + y.first;
  res.second = x.second + y.second;
  return res;
}
template<class T>
pair<T, T> operator-(pair<T,T> x, pair<T,T> y)
{
  pair<T, T> res;
  res.first = x.first - y.first;
  res.second = x.second - y.second;
  return res;
}
template<class s, class T>
pair<T,T> operator*(s ss, pair<T,T> tt)
{
  pair<T,T> res(tt);
  res.first *= ss;
  res.second *= ss;
  return res;
}
// Magnitude squared of 2D vector
template<class T>
T sq(const pair<T,T>& in)
{
  return in.first * in.first +
         in.second * in.second;
}

// Swap items in a pair of the same type
template<class T>
void swappair(pair<T,T>& p)
{
  T tmp = p.first;
  p.first = p.second;
  p.second = tmp;
}

// Some operations on vectors that are already defined
// on std::valarray, but I haven't played with that class
// yet and am on a time crunch. As a result, I'm going with
// what I know works.
template<class T>
void normalizeVector(vector<T>& vec)
{
  // Calculate L1 norm of the vector
  T norm = 0;
  for (T x: vec) norm += abs(x);
  for (T& x: vec) x /= norm;
}
template<class T>
vector<T> operator*(T fac, const vector<T>& vec)
{
  vector<T> result(vec.size());
  for (unsigned i=0; i<vec.size(); ++i)
    result[i] = fac * vec[i];
  return result;
}
template<class T>
vector<T> operator+(const vector<T>& vecl, const vector<T>& vecr)
{
  vector<T> result(vecl.size());
  for (unsigned i=0; i<vecl.size(); ++i)
    result[i] = vecl[i] + vecr[i];
  return result;
}
template<class T>
vector<T> operator*(const vector<T>& vecl, const vector<T>& vecr)
{
  vector<T> result(vecl.size());
  for (unsigned i=0; i<vecl.size(); ++i)
    result[i] = vecl[i] * vecr[i];
  return result;
}
template<class T>
void dumpVector(const vector<T>& vec, const string fname)
{
  ofstream of(fname);
  for (const T& x: vec) of << x << endl;
  of.close();
}

// Printing of pairs of stuff
template<class T, class TT>
ostream& operator<<(ostream& os, pair<T,TT> x)
{
  os << x.first << " " << x.second;
  return os;
}

/* This defines a quasirandom number generator.
 * This is code I copied from a past project, and I
 * wanted to see if this improves convergence here.
 */
class HaltonSequence                                                                   
{                                                                                      
  int base, count;                                                    
  vector<int> base_representation;                                                     
  void incseq();
                                                                                       
  public:                                                                              
    HaltonSequence(int thisbase); 
    int calc_base_digits();
    float get_xi();
    void print_seq();
};
int HaltonSequence::calc_base_digits()
{
  int n=1;
  while ((float)INT8_MAX / pow(base,n) > 1.0) ++n;
  return n;
}
HaltonSequence::HaltonSequence(int thisbase) :
  base(thisbase),
  count(1),
  base_representation(calc_base_digits(), 0)
{
  base_representation[base_representation.size()-1] = 1;
}
void HaltonSequence::incseq()
{
  count += 1;
  for (int i=base_representation.size()-1; i>=0; --i)
    if (base_representation[i] != base-1)
    {
      base_representation[i]++;
      break;
    }
    else
      base_representation[i]=0;
}
float HaltonSequence::get_xi()
{
  float res = 0.0;
  int j =1;
  for (int i=base_representation.size()-1; i>=0; --i)
    res += (float)base_representation[i] / (float)pow(base,j++);
  incseq();
  return res;
}

// --- Definition of the only quadrature set anyone will ever need ---
template<unsigned npolar>
class TabuchiYamamoto
{
  // default to a single sin theta quadrature
  const unsigned npolar_;
  const array<float, npolar> sinTheta;
  const array<float, npolar> weights;
  public:
    inline unsigned nPolar(){ return npolar_; }
    inline float getSinTheta(unsigned indx) { return sinTheta[indx]; }
    inline float getWeight(unsigned indx) { return weights[indx]; }
    explicit TabuchiYamamoto();

};
template<>
TabuchiYamamoto<1>::TabuchiYamamoto::TabuchiYamamoto() :
  npolar_(1),
  sinTheta({0.798184}),
  weights({1.0})
{
}
template<>
TabuchiYamamoto<2>::TabuchiYamamoto::TabuchiYamamoto() :
  npolar_(2),
  sinTheta({0.363900,
            0.899900}),
  weights({0.212854,
           0.787146})
{
}
template<>
TabuchiYamamoto<3>::TabuchiYamamoto::TabuchiYamamoto() :
  npolar_(3),
  sinTheta({0.166648,
            0.537707,
            0.932954}),
  weights({0.046233,
           0.283619,
           0.670148})
{
}

// --- A ray in 2D space ---
class Ray2D
{
  float x, y, phi, cosphi, sinphi;

  // Restrict phi to [0,2pi)
  void resetPhi();

  public:

    // two ways to construct
    Ray2D(float thisx, float thisy, float thisphi);
    Ray2D(Pt2D thisx, float thisphi);

    // Advance to the next point
    void advance(float newx, float newy);
    void advance(Pt2D x);

    // Reflects off a wall
    void reflect_off_normal(float phi_normal);
    float intersectXPlane(float xpos);
    float intersectYPlane(float ypos);

    // Set phi, updating cosphi and sinphi
    void setPhi(float newphi);

    // Get current 2D position
    Pt2D getPosition();

    // get unit direction
    Pt2D getCosines();

    // Check wheter the absolute value of the slope is relatively high,
    // or relatively low. True if low.
    bool hasLowSlopeValue();

    // Since a ray has an infinite line associated with it,
    // x and y should be calculable from each other. This
    // is useful in finding cartesian grid intersections.
    float y_from_x(float x);
    float x_from_y(float y);
};
Ray2D::Ray2D(float thisx, float thisy, float thisphi) :
  x(thisx),
  y(thisy),
  phi(thisphi),
  cosphi(cos(phi)),
  sinphi(sin(phi)) {}
Ray2D::Ray2D(Pt2D thisx, float thisphi) :
  x(thisx.first),
  y(thisx.second),
  phi(thisphi),
  cosphi(cos(phi)),
  sinphi(sin(phi)) {}
void Ray2D::advance(float newx, float newy) { x = newx; y = newy; }
void Ray2D::advance(Pt2D newx) { x = newx.first; y = newx.second; }
void Ray2D::resetPhi()
{
  constexpr float pi2 = 2.0 * M_PI;
  if (phi >= pi2 or phi < 0.0)
  {
    float fractional_pos = phi / pi2;
    int num_subtract = floor(fractional_pos);
    phi -= num_subtract * pi2;
  }
}
void Ray2D::reflect_off_normal(float phi_normal)
{
  float newphi = -phi - M_PI + 2.0f * phi_normal;
  setPhi(newphi);
}
void Ray2D::setPhi(float newphi)
{
  phi = newphi;
  resetPhi(); // Make sure in right interval
  cosphi = cos(phi);
  sinphi = sin(phi);
}
float Ray2D::intersectXPlane(float xpos) { return (xpos-x) / cosphi; }
float Ray2D::intersectYPlane(float ypos) { return (ypos-y) / sinphi; }
Pt2D Ray2D::getPosition() { return Pt2D(x,y); }
Pt2D Ray2D::getCosines() { return Pt2D(cosphi, sinphi); }
bool Ray2D::hasLowSlopeValue()
{
  if ( phi <= M_PI / 4.0f
       or
      (phi >= 3.0 * M_PI / 4.0f and phi <= 5.0 * M_PI / 4.0f)
       or
       phi >= 7.0f * M_PI / 4.0f
     ) return true;
  else return false;
}
float Ray2D::y_from_x(float thisx)
{
  // Handle corner case
  bool is_vertical = phi < M_PI/2.0 + EPS and phi > M_PI/2.0 - EPS;
  is_vertical |= phi < M_PI*1.5f+EPS and phi > M_PI*1.5f - EPS;
  if (is_vertical) return 0.0;

  float m = tan(phi);
  float b = y - m * x;
  return m * thisx + b;
}
float Ray2D::x_from_y(float thisy)
{
  // Handle corner case
  bool is_horizontal = phi < EPS or phi > 2. * M_PI - EPS;
  is_horizontal |= phi < M_PI*+EPS and phi > M_PI - EPS;
  if (is_horizontal) return 0.0;

  float m = 1.0f/tan(phi);
  float b = x - m * y;
  return m * thisy + b;
}


class Polygon
{
  unsigned npoints;
  unsigned pt_index;
  vector<float> x;
  vector<float> y;

  public:
    Polygon(unsigned np);
    void add_point(float x, float y);
    void add_point(Pt2D xy);
    void checkFinished();
    vector<Pt2D> getPoints();
    void draw(gdImagePtr im, float square_size);
};
Polygon::Polygon(unsigned np) :
  npoints(np),
  pt_index(0),
  x(np),
  y(np)
{
}
void Polygon::add_point(float tx, float ty)
{
  x[pt_index] = tx;
  y[pt_index++] = ty;
}
void Polygon::add_point(pair<float,float> xy)
{
  x[pt_index] = xy.first;
  y[pt_index++] = xy.second;
}
void Polygon::checkFinished()
{
  if (pt_index != npoints)
  {
    cerr << "Failed to fully construct polygon before use." << endl
         << pt_index + 1 << " points were added." << endl;
    exit(1);
  }
}
vector<pair<float,float>> Polygon::getPoints()
{
  checkFinished();
  vector<pair<float,float>> res(x.size());
  int i = 0;
  for (auto &p : res)
  {
    p.first = x[i];
    p.second= y[i++];
  }
  return res;
}
void Polygon::draw(gdImagePtr im, float square_size)
{
  checkFinished();

  // Convert list of real coordinates to pixel coordinates
  std::vector<unsigned> xp(x.size());
  std::vector<unsigned> yp(x.size());

  unsigned n_pix = gdImageSX(im);
  float dx = square_size / n_pix;
  for (unsigned i=0; i<x.size(); ++i)
  {
    int x0, y0;
    x0 = (x[i] + square_size / 2.0) / dx;
    y0 = (y[i] + square_size / 2.0) / dx;
    // coodinates are from top left out
    xp[i] = x0;
    yp[i] = n_pix - y0;
  }

  int black = gdImageColorAllocate(im, 0, 0, 0);
  unsigned i=0;
  while (i<x.size()-1)
  {
    gdImageLine(im, xp[i], yp[i], xp[i+1], yp[i+1], black);
    i++;
  }
  gdImageLine(im, xp[0], yp[0], xp[i], yp[i], black);
}

// Stores run settings
struct RunSettings
{
  float ray_length, deadzone_length;
  string xslibrary;
  unsigned mesh_dimx;
  unsigned ngroups;
  unsigned raysperiteration;
  unsigned ninactive;
  unsigned nactive;
  unsigned npolar;

  public:
    RunSettings(string inputfile);
};
RunSettings::RunSettings(string inputfile)
{
  ifstream instream(inputfile);
  string word;
  while (instream >> word)
  {
    if (word == "deadlength")
      instream >> deadzone_length;
    else if (word == "raylength")
      instream >> ray_length;
    else if (word == "xslibrary")
      instream >> xslibrary;
    else if (word == "mesh_dimx")
      instream >> mesh_dimx;
    else if (word == "ngroups")
      instream >> ngroups;
    else if (word == "raysperiteration")
      instream >> raysperiteration;
    else if (word == "nactive")
      instream >> nactive;
    else if (word == "ninactive")
      instream >> ninactive;
    else if (word == "npolar")
      instream >> npolar;
  }
  instream.close();
}

// Holds all of the macroscopic cross sections needed for a steady-state flux solution
struct Material
{
  string name;
  unsigned ngroups;
  bool fissile;

  vector<float> trans, abs, nuscat, chi, nufiss;

  static const array<const string, 3> xs_types;
  static const array<const string, 2> fiss_xs_types;

  public:
    Material(unsigned ngroups, bool fissile = false);
    void setFissile();
};
const array<const string, 3> Material::xs_types = {"trans", "abs", "nuscat"} ;
const array<const string, 2> Material::fiss_xs_types = {"chi", "nufiss"};
Material::Material(unsigned thisngroups, bool thisfissile) :
  ngroups(thisngroups),
  fissile(thisfissile),
  trans(ngroups),
  abs(ngroups),
  nuscat(ngroups*ngroups),
  chi(fissile ? ngroups : 0),
  nufiss(fissile ? ngroups : 0)
{
}
void Material::setFissile()
{
  fissile = true;
  chi.resize(ngroups);
  nufiss.resize(ngroups);
}

// --- Implements the actual random ray algorithm ---
template<class G, class M, class Q>
class Solver2D
{
  G geometry; // has to give next FSR ID given a ray
  M materialSet; // has to return cross sections given an FSR ID
  Q polarQuadrature;
  RunSettings settings;

  unsigned ngroups;
  unsigned nfsr;
  vector<float> fluxes;
  vector<float> fiss_source;
  vector<float> scat_source;
  vector<float> cell_distance_traveled;

  void calcSource();
  void transportSweep();

  /* These hold the cell ID numbers, and length traversed across
   * each cell after a ray has been moved from a point to an edge */
  unsigned max_fsr_crossings;
  vector<int> fsr_crossing_ids;
  vector<float> fsr_crossing_lengths;

  public:
    Solver2D(G geom, M materials, RunSettings setts);
    float calcEigenvalue();

    /* run a ray starting from this given x0 and phi */
    void run_ray(Pt2D x0, float phi, unsigned ray_id=0);

    /* reference to geometry, for plotting and the like */
    G& getGeom();

    /* Set fission source at a single point */
    void setSource(unsigned group, unsigned fsr_id, float value);

    /* Calculate scattering source from current flux */
    void scatter();
    float fission(float k);
    void zeroFlux();

    // for dumping fields to files
    void dumpFluxes(string fname);
    void dumpScatter(string fname);
    void dumpFission(string fname);

    void normalizeFlux();
    void normalizeByRelativeTraversalDistance();
    void multiplyFlux(float x);
    void addSourceToScalarFlux();

    const vector<float>& getFlux() const;
};
template<class G, class M, class Q>
Solver2D<G, M, Q>::Solver2D(G geom, M materials, RunSettings setts) :
  geometry(geom),
  materialSet(materials),
  settings(setts),
  ngroups(setts.ngroups),
  nfsr(setts.mesh_dimx * setts.mesh_dimx),
  fluxes(setts.mesh_dimx * setts.mesh_dimx * ngroups, 1.0f),
  fiss_source(fluxes.size(), 0.0f),
  scat_source(fluxes.size(), 0.0f),
  cell_distance_traveled(nfsr, 0.0f),
  /* Calculate maximum possible number of FSRs that can be crossed as a ray
   * goes from one boundary to the next. This list terminates with a -1 cell
   * ID. */
  max_fsr_crossings(2 * settings.mesh_dimx),
  fsr_crossing_ids(max_fsr_crossings)
{
}
template <class G, class M, class Q>
const vector<float>& Solver2D<G, M, Q>::getFlux() const { return fluxes; }

template <class G, class M, class Q>
void Solver2D<G, M, Q>::normalizeFlux()
{
  float sum_flux = 0.0;

  // Loop over all fluxes
  for (unsigned fsr=0; fsr<nfsr; ++fsr)
  {
    for (unsigned g=0; g<ngroups; ++g)
    {
      sum_flux += abs(fluxes[ngroups * fsr + g]);
    }
  }
  if (sum_flux == 0.0f)
  {
    cerr << "Got zero flux when attempting to normalize flux." << endl;
    exit(1);
  }
  transform(fluxes.begin(), fluxes.end(), fluxes.begin(),
      bind(multiplies<float>(), placeholders::_1, 1.0f/sum_flux));
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::normalizeByRelativeTraversalDistance()
{
  for (unsigned fsr=0; fsr<nfsr; ++fsr)
    for (unsigned g=0; g<ngroups; ++g)
      fluxes[fsr*ngroups + g] /= cell_distance_traveled[fsr];
  fill(cell_distance_traveled.begin(), cell_distance_traveled.end(), 0.0f);
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::zeroFlux()
{
  fill(fluxes.begin(), fluxes.end(), 0.0f);
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::addSourceToScalarFlux()
{
  for (unsigned fsr=0; fsr<nfsr; ++fsr)
  {
    string mat_name;
    if (geometry.inside_fuel(fsr))
      mat_name = "fuel";
    else
      mat_name = "mod";
    const Material& mat = materialSet.getMaterial(mat_name);
    const vector<float>& sigmat = mat.trans;
    for (unsigned g=0; g<ngroups; ++g)
    {
      unsigned indx = ngroups * fsr + g;
      fluxes[indx] += (fiss_source[indx] + scat_source[indx]) / sigmat[g];
    }
  }
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::multiplyFlux(float x)
{
  for (auto& f : fluxes) f *= x;
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::setSource(unsigned group, unsigned fsr_id, float value)
{
  unsigned flux_id = ngroups * fsr_id + group;
  fiss_source[flux_id] = value;
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::scatter()
{
  for (unsigned fsr=0; fsr<nfsr; ++fsr)
  {
    string mat_name;
    if (geometry.inside_fuel(fsr))
      mat_name = "fuel";
    else
      mat_name = "mod";
    const Material& mat = materialSet.getMaterial(mat_name);
    const vector<float>& scatmat = mat.nuscat;
    for (unsigned g=0; g<ngroups; ++g)
    {
      scat_source[ngroups * fsr + g] = 0.0f;
      for (unsigned gprime=0; gprime<ngroups; ++gprime)
      {
        scat_source[ngroups * fsr + g] += 
          scatmat[g*ngroups + gprime] * fluxes[ngroups * fsr + gprime];
      }
    }
  }
}
template <class G, class M, class Q>
float Solver2D<G, M, Q>::fission(float k)
{
  float fissionSource = 0.0;
  for (unsigned fsr=0; fsr<nfsr; ++fsr)
  {
    string mat_name;
    if (geometry.inside_fuel(fsr))
      mat_name = "fuel";
    else
    {
      mat_name = "mod";
      continue;
    }
    const Material& mat = materialSet.getMaterial(mat_name);
    const vector<float>& nusigf = mat.nufiss;
    const vector<float>& chi = mat.chi;
    // NOTE could be done more efficiently
    for (unsigned g=0; g<ngroups; ++g)
    {
      fiss_source[ngroups * fsr + g] = 0.0f;
      for (unsigned gprime=0; gprime<ngroups; ++gprime)
      {
        float this_fiss = chi[g] * fluxes[ngroups * fsr + gprime] * nusigf[gprime] / k;
        fiss_source[ngroups * fsr + g] += this_fiss;
        fissionSource += this_fiss;
      }
    }
  }
  return fissionSource;
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::run_ray(Pt2D x0, float phi, unsigned ray_id)
{
  Ray2D ray(x0, phi);
  float total_distance_remaining = settings.ray_length;

  float dead_length_remaining = settings.deadzone_length;
  float dist;
  bool live = false;
  unsigned npolar = polarQuadrature.nPolar();
  constexpr float pi4 = 4.0f * M_PI;
  float live_length = settings.ray_length - settings.deadzone_length;

  // NOTE could give speed boost with std::array here
  vector<float> track_fluxes(ngroups * npolar, 0.0);

  // Loop over dead length (comments are in the live loop)
  while (dead_length_remaining > 0.0)
  {
    dist = geometry.advance_to_boundary(ray, fsr_crossing_ids, fsr_crossing_lengths, ray_id);
    for (unsigned s=0; s<fsr_crossing_ids.size(); ++s)
    {
      int fsr_id = fsr_crossing_ids[s];
      float segment_length = fsr_crossing_lengths[s];
      dead_length_remaining -= segment_length;
      total_distance_remaining -= segment_length;
      if (segment_length == 0.0f) continue;
      string matname;
      if (geometry.inside_fuel(fsr_id))
        matname = "fuel";
      else
        matname = "mod";
      const vector<float>& sigmat = materialSet.getMaterial(matname).trans;
      for (unsigned g=0; g<ngroups; ++g) // groups
        for (unsigned p=0; p<npolar; ++p) // polars
        {
          int scalar_flux_index = ngroups * fsr_id + g;
          float tau = segment_length / polarQuadrature.getSinTheta(p) * sigmat[g];
          float delta_psi = (track_fluxes[p+g*npolar] -
                            (fiss_source[scalar_flux_index] +
                             scat_source[scalar_flux_index])/(pi4*sigmat[g]))*
                            (1.0f - expf(-tau));
          track_fluxes[p+g*npolar] -= delta_psi;
        }
    }
  }

  // Loop over live length
  while (total_distance_remaining > 0.0)
  {
    dist = geometry.advance_to_boundary(ray, fsr_crossing_ids, fsr_crossing_lengths, ray_id);

    for (unsigned s=0; s<fsr_crossing_ids.size(); ++s) // segments
    {
      int fsr_id = fsr_crossing_ids[s];
      float segment_length = fsr_crossing_lengths[s];
      total_distance_remaining -= segment_length;
      if (segment_length == 0.0f) continue;
      string matname;
      if (geometry.inside_fuel(fsr_id))
        matname = "fuel";
      else
        matname = "mod";

      const vector<float>& sigmat = materialSet.getMaterial(matname).trans;

      for (unsigned g=0; g<ngroups; ++g) // groups
        for (unsigned p=0; p<npolar; ++p) // polars
        {
          int scalar_flux_index = ngroups * fsr_id + g;
          float tau = segment_length / polarQuadrature.getSinTheta(p) * sigmat[g];
          float delta_psi = (track_fluxes[p+g*npolar] -
                            (fiss_source[scalar_flux_index] +
                             scat_source[scalar_flux_index])/(pi4*sigmat[g]))*
                            (1.0f - expf(-tau));
          fluxes[scalar_flux_index] += delta_psi / sigmat[g] * polarQuadrature.getWeight(p);
          track_fluxes[p+g*npolar] -= delta_psi;
        }
      for (unsigned p=0; p<npolar; ++p)
        cell_distance_traveled[fsr_id] += segment_length / polarQuadrature.getSinTheta(p)
           * polarQuadrature.getWeight(p);
    }
  }
}
template <class G, class M, class Q>
G& Solver2D<G, M, Q>::getGeom() { return geometry; }

template <class G, class M, class Q>
void Solver2D<G, M, Q>::dumpFluxes(string fname)
{
  ofstream f(fname, ofstream::out);
  for (auto flx : fluxes) f << flx << endl;
  f.close();
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::dumpScatter(string fname)
{
  ofstream f(fname, ofstream::out);
  for (auto flx : scat_source) f << flx << endl;
  f.close();
}
template <class G, class M, class Q>
void Solver2D<G, M, Q>::dumpFission(string fname)
{
  ofstream f(fname, ofstream::out);
  for (auto flx : fiss_source) f << flx << endl;
  f.close();
}

// Used for any scenario when only a finite amount of materials exist in
// a problem. This is anything without depletion, pretty much.
class FiniteMaterialSet
{
  unsigned nmaterials;
  vector<Material> materials;
  map<string, unsigned> material_map;

  void loadVector(vector<float>& to_vec, fs::path infile);
  static unsigned getMaterialCount(string libname);
  public:

    const Material& getMaterial(string name);

    FiniteMaterialSet(string xslib, unsigned ngroups);
};
void FiniteMaterialSet::loadVector(vector<float>& to_vec, fs::path infile)
{
  // Checks correct number XS loaded
  unsigned loadCount = 0;
  ifstream instream(infile, ifstream::in);
  if (not instream.good())
  {
    cerr << "cannot load " << infile << endl;
    exit(1);
  }
  float value;
  while (instream >> value)
  {
    to_vec[loadCount++] = value;
    if (loadCount > to_vec.size())
    {
      cerr << "Tried to load too many XS from material " << infile << endl;
      cerr << "too many groups or too few?" << endl;
      exit(1);
    }
  }
  if (loadCount != to_vec.size())
  {
    cerr << "too few xs values in " << infile << endl;
    exit(1);
  }
}
FiniteMaterialSet::FiniteMaterialSet(string xslib, unsigned ngroups) :
  nmaterials(getMaterialCount(xslib)),
  materials(nmaterials, Material(ngroups))
{
  if (nmaterials == 0)
  {
    cout << "zero materials were found in xslib named: " << xslib << endl;
    exit(1);
  }

  unsigned mat_indx = 0;
  fs::path p(xslib);
  for (const auto& entry : fs::directory_iterator(p))
  {
    // Load required XS
    string materialname = entry.path().filename();
    cout << "Processing material " << materialname << endl;
    Material& mat = materials[mat_indx];
    for (string xs_type : Material::xs_types)
    {
      loadVector(mat.trans, entry/"trans");
      loadVector(mat.abs, entry/"abs");
      loadVector(mat.nuscat, entry/"nuscat");
    }

    // Maybe load fissile XS
    vector<bool> fissile_xs_present(Material::fiss_xs_types.size(), false);
    unsigned fiss_i = 0;
    for (string fiss_xs : Material::fiss_xs_types)
      if (fs::exists(entry/fiss_xs)) fissile_xs_present[fiss_i++] = true;
    bool no_fiss = none_of(fissile_xs_present.begin(), fissile_xs_present.end(),
        [](bool x){ return x; });
    bool all_fiss = any_of(fissile_xs_present.begin(), fissile_xs_present.end(),
        [](bool x){ return x; });
    if (not no_fiss ^ all_fiss) { cerr << "Some, but not all fiss. XS found. " << endl; exit(1); }
    if (all_fiss)
    {
      mat.setFissile();
      for (string fiss_xs : Material::fiss_xs_types)
      {
        loadVector(mat.chi, entry/"chi");
        loadVector(mat.nufiss, entry/"nufiss");
      }
    }

    pair<string, unsigned> mat_dict_entry(materialname, mat_indx++);
    material_map.insert(mat_dict_entry);
  }

}
const Material& FiniteMaterialSet::getMaterial(string name) { return materials[material_map[name]]; }
unsigned FiniteMaterialSet::getMaterialCount(string libname)
{
  fs::path p(libname);
  string filename;
  unsigned nmaterials = 0;

  // check all required cross sections present in each material
  for (const auto& entry : fs::directory_iterator(p))
  {
    if (fs::is_directory(entry))
    {
      for (string xs_type : Material::xs_types)
        if (not fs::exists(entry/xs_type))
        {
          cerr << "Required cross section " << xs_type <<
            " not found in material " << entry << endl;
          exit(1);
        }
      ++nmaterials;
    }
    else
    {
      cerr << "Found non-directory file in xslib." << endl;
      cerr << "That shouldn't be there. Name: " << entry << endl;
      exit(1);
    }
  }
  return nmaterials;
}

// --- A geometry class for the pedagogical fuel assembly in 22.212 ---
struct SquarePinGeom
{
  unsigned mesh_dimx;
  float mesh_dx;
  array<unsigned, 6> index_endpoints;

  // Prescribed fuel dimensions:
  static constexpr float pitch = 1.2;
  static constexpr float assembly_width = 3.0 * pitch;
  static constexpr float assembly_radius = assembly_width / 2.0;
  static constexpr float pin_width = pitch / 3.0;

  // Helper for ray intersections with cartesian mesh
  void boundIndex(int& i);
  pair<int,int> roundRayEndpoints(pair<float,float> ray_dxf);

  // Tell whether a point is between two other points
  // on either side, with the sign of the points on either
  // side not pre-determined:
  bool isSandwiched(float lo, float x, float hi);

  // get FSR ID from xy coordinates
  int fsr_from_xy(Pt2D& pt);

  public:
    SquarePinGeom(unsigned mesh_dimx);
    bool inside_fuel(unsigned i, unsigned j);
    bool inside_fuel(unsigned indx);
    std::vector<Polygon> get_pin_boundaries();

    // Move ray to next mesh intersection, and return distance traveled
    float advance_to_boundary(Ray2D& ray, vector<int>& fsrs_crossed,
                              vector<float>& cross_lengths,
                              unsigned ray_id=0);
};
SquarePinGeom::SquarePinGeom(unsigned mesh_dimx) :
  mesh_dimx(mesh_dimx),
  mesh_dx(assembly_width / (float)mesh_dimx)
{
  if (mesh_dimx % 9 != 0)
  {
    cerr << "Mesh cell count in one direction must be divisible by 9. Got "
         << mesh_dimx << endl;
    exit(1);
  }
  unsigned fuelwide = mesh_dimx / 9;
  for (unsigned i=0; i<3; ++i)
  {
    index_endpoints[2*i] = fuelwide + i * mesh_dimx / 3;
    index_endpoints[2*i+1] = index_endpoints[2*i] + fuelwide - 1;
  }
}
bool SquarePinGeom::inside_fuel(unsigned i, unsigned j)
{
  // Loop over all 6 pin edge corner indices
  bool i_inside = false;
  bool j_inside = false;
  bool i_centered = false;
  bool j_centered = false;
  for (unsigned indx=0; indx < 3; ++indx)
  {
    if (i <= index_endpoints[2*indx+1] and i >= index_endpoints[2*indx]) i_inside=true;
    if (j <= index_endpoints[2*indx+1] and j >= index_endpoints[2*indx]) j_inside=true;
  }
  if (i <= index_endpoints[3] and i >= index_endpoints[2]) i_centered=true;
  if (j <= index_endpoints[3] and j >= index_endpoints[2]) j_centered=true;

  return (i_inside and j_inside) and not (i_centered and j_centered);
}
bool SquarePinGeom::isSandwiched(float lo, float x, float hi)
{
  bool a = lo <= x;
  bool b = x <= hi;
  // M'logic
  return (a and b) or not (a or b);
}
bool SquarePinGeom::inside_fuel(unsigned indx)
{
  // Calculate discrete cartesian coordinates
  unsigned i = indx / mesh_dimx;
  unsigned j = indx % mesh_dimx;
  return inside_fuel(i, j);
}
vector<Polygon> SquarePinGeom::get_pin_boundaries()
{
  vector<Polygon> result(8,4);

  // Pin positions in units of pitch
  vector<int> x_pit = {-1, 0, 1, -1, 1, -1, 0, 1};
  vector<int> y_pit = {-1, -1, -1, 0, 0, 1, 1, 1};

  // corner masks
  vector<int> maskx = {-1, 1, 1, -1};
  vector<int> masky = {-1, -1, 1, 1};

  for (unsigned i=0; i<result.size(); ++i)
  {
    pair<float,float> center;
    center.first = x_pit[i] * pitch;
    center.second = y_pit[i] * pitch;

    // add pin corners:
    for (int j=0; j<4; ++j)
    {
      pair<float,float> u(maskx[j], masky[j]);
      result[i].add_point(center + (pin_width / 2.0f) * u);
    }
  }
  return result;
}
int SquarePinGeom::fsr_from_xy(Pt2D& pt)
{
  int ix, iy;
  // Need to handle points which lie very slightly or too close to
  // the problem boundary
  if (pt.first > assembly_radius-EPS) pt.first = assembly_radius-EPS;
  if (pt.first < -assembly_radius+EPS) pt.first = -assembly_radius+EPS;
  if (pt.second > assembly_radius-EPS) pt.second = assembly_radius-EPS;
  if (pt.second < -assembly_radius+EPS) pt.second = -assembly_radius+EPS;
  ix = (int) ((pt.first + assembly_radius) / mesh_dx);
  iy = (int) ((pt.second + assembly_radius) / mesh_dx);
  return ix + mesh_dimx * iy;
}
float SquarePinGeom::advance_to_boundary(Ray2D& ray, vector<int>& fsrs_crossed,
    vector<float>& cross_lengths, unsigned ray_id)
{
  // Open up tracks output file
  // ofstream o("tracks", ofstream::app);

  // Signed distances to boundary. Positive => along right direction
  array<float, 4> dvalues;

  // Loop over boundaries (east, north, west, south)
  dvalues[0] = ray.intersectXPlane(assembly_radius);
  dvalues[1] = ray.intersectYPlane(assembly_radius);
  dvalues[2] = ray.intersectXPlane(-assembly_radius);
  dvalues[3] = ray.intersectYPlane(-assembly_radius);

  // Find the surface that actually got intersected:
  float dmin = HUGE_VALF; // -> valf sounds like a D&D creature
  int indx = -1;
  
  // Save old ray position
  Pt2D oldpos = ray.getPosition();
  // o << oldpos << " " << ray_id << endl;

  // Find surface intersection
  for(int i=0; i<4; ++i)
    if(dvalues[i] < dmin and dvalues[i] > EPS) 
    {
      indx = i;
      dmin = dvalues[i];
    }

  if (indx == -1 or dmin > 2. * assembly_width) 
  {
    cerr << "No surface intersection found." << endl;
    cerr << "Ray info:" << endl;
    cerr << "    Origin = " << ray.getPosition()<< endl;
    cerr << "    cosines = " << ray.getCosines() << endl;
    exit(1);
  }

  // Calc new ray position
  Pt2D cos = ray.getCosines();
  Pt2D newpos = oldpos + dmin * cos;

  // Find non-integer coordinates along cartesian mesh
  float min_xcross_f = (oldpos.first + assembly_radius) / mesh_dx;
  float min_ycross_f = (oldpos.second + assembly_radius) / mesh_dx;
  float max_xcross_f = (newpos.first + assembly_radius) / mesh_dx;
  float max_ycross_f = (newpos.second + assembly_radius) / mesh_dx;

  /* Determine range of indices where a ray intersection was encountered */
  pair<int,int> cross_bounds_x, cross_bounds_y;
  cross_bounds_x = roundRayEndpoints(pair<float,float>(min_xcross_f, max_xcross_f));
  cross_bounds_y = roundRayEndpoints(pair<float,float>(min_ycross_f, max_ycross_f));

  // Crossing points on the mesh
  int n=0;
  unsigned max_num_crossings(abs(max_ycross_f-min_ycross_f) +
                             abs(max_xcross_f-min_xcross_f) + 5);
  vector<Pt2D> mesh_crossings(max_num_crossings);

  /* Count number of crossings bounded by ray endpoints */
  if (cross_bounds_x.first > cross_bounds_x.second) swappair(cross_bounds_x);
  if (cross_bounds_y.first > cross_bounds_y.second) swappair(cross_bounds_y);

  /* Loop over possible x values and check if they're inside the ray */
  for (int xi=cross_bounds_x.first; xi<=cross_bounds_x.second; ++xi)
  {
    float thisx = xi * mesh_dx - assembly_radius;
    float thisy = ray.y_from_x(thisx);
    Pt2D pt(thisx, thisy);
    if (isSandwiched(oldpos.first, thisx, newpos.first)) mesh_crossings[n++] = pt;
  }
  /* Loop over possible y values */
  for (int yi=cross_bounds_y.first; yi<=cross_bounds_y.second; ++yi)
  {
    float thisy = yi * mesh_dx - assembly_radius;
    float thisx = ray.x_from_y(thisy);
    Pt2D pt(thisx, thisy);
    if (isSandwiched(oldpos.first, thisx, newpos.first)) mesh_crossings[n++] = pt;
  }

  // Initially overestimated count of mesh crossings:
  mesh_crossings.resize(n);

  // Only interior line crossings were calculated, so num_segments=n+1
  fsrs_crossed.resize(n+1);
  cross_lengths.resize(n+1);

  // Sort FSR crossings distances based on distance from ray origin:
  sort(mesh_crossings.begin(), mesh_crossings.end(),
       [oldpos](const Pt2D& a, const Pt2D& b) -> bool
       { return sq(a-oldpos) < sq(b-oldpos); });

  // Fill out FSR crossings and respective lengths. This is not done in
  // an optimally fast way, but instead in one that I know will be very
  // safe when using single precision numbers.
  Pt2D midpoint;

  // First FSR
  cross_lengths[0] = pow(sq(mesh_crossings[0]-oldpos), 0.5f);
  midpoint = 0.5f * (mesh_crossings[0] + oldpos);
  fsrs_crossed[0] = fsr_from_xy(midpoint);

  // In-between FSRs
  for (int s=1; s<n; ++s)
  {
    cross_lengths[s] = pow(sq(mesh_crossings[s]-mesh_crossings[s-1]), 0.5f);
    midpoint = 0.5f * (mesh_crossings[s]+mesh_crossings[s-1]);
    fsrs_crossed[s] = fsr_from_xy(midpoint);
  }

  // Last FSR
  cross_lengths[n] = pow(sq(newpos-mesh_crossings[n-1]), 0.5f);
  midpoint = 0.5f * (newpos+mesh_crossings[n-1]);
  fsrs_crossed[n] = fsr_from_xy(midpoint);

  ray.advance(newpos);
  ray.reflect_off_normal(M_PI + indx * M_PI_2);

  // o.close();

  return dmin;
}
void SquarePinGeom::boundIndex(int& x)
{
  if (x < 0) x = 0;
  else if ((unsigned)x > mesh_dimx) x = mesh_dimx - 1;
}
pair<int, int> SquarePinGeom::roundRayEndpoints(pair<float, float> ray_dxf)
{
  pair<int, int> result;
  if (ray_dxf.first > ray_dxf.second)
  {
    // Round min up
    result.first = ceil(ray_dxf.first);
    result.second = floor(ray_dxf.second);
  }
  else
  {
    // Ray is traveling right; round down
    result.first = floor(ray_dxf.first);
    result.second = ceil(ray_dxf.second);
  }
  return result;
}
// Requires libgd. Makes a square picture of the geometry.
template<class T>
gdImagePtr plot_geometry(T& geometry, unsigned npixel)
{
  gdImagePtr im;
  string filename;
  int black, white, grey;

  ostringstream fname;
  fname << "geometry_"<< npixel << ".png";
  filename = fname.str();

  im = gdImageCreate(npixel, npixel);
  white = gdImageColorAllocate(im, 255, 255, 255);
  black = gdImageColorAllocate(im, 0, 0, 0);
  grey = gdImageColorAllocate(im, 128, 128, 128);

  // draw each pin
  vector<Polygon> pins = geometry.get_pin_boundaries();
  for (auto p : pins)
    p.draw(im, SquarePinGeom::assembly_width);

  // Draw cartesian mesh divisions:
  float dx = SquarePinGeom::assembly_width / npixel;
  float mesh_dx = SquarePinGeom::assembly_width / geometry.mesh_dimx;
  for (unsigned i=0; i<geometry.mesh_dimx; ++i)
  {
    int x0 = mesh_dx * i / dx;
    gdImageDashedLine(im, x0, 0, x0, npixel, grey);
    gdImageDashedLine(im, 0, x0, npixel, x0, grey);
  }

  return im;
}

/* Return a point in a box of given radius */
pair<float, float> random_point(float rad, float xi1, float xi2)
{
  pair<float, float> res;
  float xmax = rad;
  float xmin = -rad;
  float ymax = rad;
  float ymin = -rad;
  res.first = xi1 * (xmax - xmin) + xmin;
  res.second = xi2 * (ymax - ymin) + ymin;
  return res;
}
pair<float, float> random_point(float rad)
{
  float xi1 = randf();
  float xi2 = randf();
  return random_point(rad, xi1, xi2);
}

int main(int argc, char* argv[]) 
{ 
  // Check command line input
  if (argc != 2)
  {
    cerr << "Should pass a settings file as a command line argument" << endl;
    exit(1);
  }
  string filename = argv[1];

  // create solver object. Could do some rvalue reference stuff
  // here, but speed isn't really a concern right here.
  RunSettings settings(filename);
  FiniteMaterialSet materials(settings.xslibrary, settings.ngroups);
  SquarePinGeom geom = SquarePinGeom(settings.mesh_dimx);

  Solver2D<SquarePinGeom,
    FiniteMaterialSet,
    TabuchiYamamoto<NPOLAR>> solver(geom,
                               materials,
                               settings);

  // plot geometry, and return an image pointer for plotting tracks over it
  gdImagePtr thisim;
  thisim = plot_geometry(solver.getGeom(), 900);

  // Delete old track file
  remove("tracks");

  // Check that npolar given in input matches compile time value
  if (NPOLAR != settings.npolar)
  {
    cerr << "recompile if you want to change npolar" << endl;
    exit(1);
  }

  // Set point source for testing (group 0 flux in FSR 364).
  // This gets the iterations moving.
  // solver.setSource(0, 364, 1.0f);

  // Quasirandom number generators:
  // They improve convergence of low dimensional Monte Carlo simulations.
  // HaltonSequence x1(2);
  // HaltonSequence x2(3);
  // HaltonSequence x3(4);
  // for (int i=0; i<7; ++i) x1.get_xi();
  // for (int i=0; i<43; ++i) x2.get_xi();
  // for (int i=0; i<13; ++i) x3.get_xi();

  unsigned nrays = settings.raysperiteration;
  float k = 1.0f;
  float oldFissionSource=1.0f;
  float fissionSource=1.0f;

  // Set up flux guess
  solver.normalizeFlux();

  // Play cool animation while iterating
  vector<const char*> symbols = {"  |>-----|",
                                 "  |->----|",
                                 "  |-->---|",
                                 "  |--->--|",
                                 "  |---->-|",
                                 "  |----->|",
                                 "  |-----<|",
                                 "  |----<-|",
                                 "  |---<--|",
                                 "  |--<---|",
                                 "  |-<----|",
                                 "  |<-----|"};
  // Loop over inactive iterations
  for (unsigned n=0; n<settings.ninactive; ++n)
  {
    solver.scatter();
    oldFissionSource = fissionSource;
    fissionSource = solver.fission(k);
    k *= fissionSource / oldFissionSource;
    cout << "\r" << "Inactive iteration (" << setw(5) << n+1 << "/"
      << setw(5) << settings.ninactive
      << ") " << "k = " << setprecision(7) << scientific << k
      << symbols[n % symbols.size()] << flush;
    solver.zeroFlux();

    // Loop over all rays
    for (unsigned i=0; i<nrays; ++i)
    {
      Pt2D point = random_point(solver.getGeom().assembly_radius);
      float angle = randf() * M_PI * 2.0f;
      solver.run_ray(point, angle, i);
    }

    solver.normalizeByRelativeTraversalDistance();
    solver.multiplyFlux(M_PI * 4.0f); // Done after the fact
    solver.addSourceToScalarFlux();
  }
  cout << endl << "Inactive iterations successfully completed." << endl;

  // Loop over active iterations
  float k_avg = 0.0;
  float k_sq_avg = 0.0;
  vector<float> tmp_flux;
  vector<float> flux_result(solver.getFlux().size());
  vector<float> flux_sq(solver.getFlux().size());
  for (unsigned n=0; n < settings.nactive; ++n)
  {
    solver.scatter();
    oldFissionSource = fissionSource;
    fissionSource = solver.fission(k);
    k *= fissionSource / oldFissionSource;
    k_avg = (n * k_avg + k) / (n+1.0f);
    k_sq_avg = (n * k_sq_avg + k*k) / (n+1.0f);
    float this_sigma = sqrt(1.0f/(n-1.f) * (k_sq_avg - k_avg*k_avg));
    float uncert = this_sigma / sqrt(n);
    cout << "\r" << "Active iteration (" << setw(5) << n+1 << "/"
      << setw(5) << settings.nactive
      << ") " << "k = " << setprecision(7) << scientific << k_avg
      << " +/- " << setprecision(3) << scientific << uncert / k * 100.0f << "%"
      << symbols[n % symbols.size()] << flush;
    solver.zeroFlux();

    // Loop over all rays
    for (unsigned i=0; i<nrays; ++i)
    {
      Pt2D point = random_point(solver.getGeom().assembly_radius);
      float angle = randf() * M_PI * 2.0f;
      solver.run_ray(point, angle, i);
    }

    solver.normalizeByRelativeTraversalDistance();
    solver.multiplyFlux(M_PI * 4.0f); // Done after the fact
    solver.addSourceToScalarFlux();

    // Aggregate stochastic fluxes
    tmp_flux = solver.getFlux();
    normalizeVector(tmp_flux);
    flux_result = 1.0f/(n+1.0f)*((float)n * flux_result + tmp_flux);
    flux_sq = 1.0f/(n+1.0f)*((float)n * flux_sq + tmp_flux*tmp_flux);
  }
  cout << endl;

  dumpVector(flux_result, "flux.out");
  dumpVector(flux_sq, "fluxsq.out");

  FILE *pngout;
  string picFileName = "geometry.png";
  pngout = fopen(picFileName.c_str(), "wb");
  gdImagePng(thisim, pngout);
  fclose(pngout);
  gdImageDestroy(thisim);
}
