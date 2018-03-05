/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 10;
  particles.reserve(num_particles);
  weights.reserve(num_particles);

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i=0; i < num_particles; i++) {
    particles.emplace_back(
        Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0 }
    );
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(auto &p: particles) {
    if (fabs(yaw_rate) < EPSILON) {
      p.x += delta_t * velocity * cos(p.theta);
      p.y += delta_t * velocity * sin(p.theta);
    } else {
      double v_over_yaw_rate = velocity / yaw_rate;
      double yaw_change = yaw_rate * delta_t;
      double xf = p.x + v_over_yaw_rate * (sin(p.theta + yaw_change) - sin(p.theta));
      double yf = p.y + v_over_yaw_rate * (cos(p.theta) - cos(p.theta + yaw_change));
      double thetaf = p.theta + yaw_change;

      p.x = xf;
      p.y = yf;
      p.theta = thetaf;
    }

    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for(auto &obs: observations) {
    auto min = min_element(predicted.begin(), predicted.end(),
        [&obs] (const LandmarkObs &l1, const LandmarkObs &l2) {
          return obs.distance(l1) < obs.distance(l2);
        });
    if (min != predicted.end()) {
      obs.id = min->id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  weights.clear();
  auto guass_norm = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  for(auto &p: particles) {

    std::vector<LandmarkObs> landmarks;
    for(auto lm: map_landmarks.landmark_list) {
      if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range) {
        landmarks.emplace_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }

    std::vector<LandmarkObs> mapObservations = observations;
    convertToMapCoordinates(p, mapObservations);
    dataAssociation(landmarks, mapObservations);

    std::vector<double> weight_accumulator;
    for(auto obs: mapObservations) {
      auto closestLandmark = getAssociatedLandmark(obs, landmarks);
      if (closestLandmark != landmarks.end()) {
        auto two_sig_x_sq = 2 * pow(std_landmark[0], 2);
        auto two_sig_y_sq = 2 * pow(std_landmark[1], 2);
        auto xdiff = obs.x - closestLandmark->x;
        auto ydiff = obs.y - closestLandmark->y;
        auto exponent = (pow(xdiff,2) / two_sig_x_sq) + (pow(ydiff,2) / two_sig_y_sq);
        auto weight = guass_norm * std::exp(-exponent);
        weight_accumulator.emplace_back(weight);
      }
    }
    p.weight = std::accumulate(weight_accumulator.begin(), weight_accumulator.end(), 1.0, std::multiplies<double>());
    weights.emplace_back(p.weight);
  }
}

std::vector<const LandmarkObs>::iterator ParticleFilter::getAssociatedLandmark(const LandmarkObs &obs,const std::vector<LandmarkObs> &landmarks) {
  int id = obs.id;
  auto result = std::find_if(landmarks.begin(), landmarks.end(), [id] (LandmarkObs const &lm) {
      return id == lm.id;
      });
  return result;
}

void ParticleFilter::convertToMapCoordinates(const Particle &p, std::vector<LandmarkObs> &observations) {
  double sin_theta = sin(p.theta);
  double cos_theta = cos(p.theta);
  for(auto &obs: observations) {
    auto map_x = p.x + (cos_theta * obs.x) - (sin_theta * obs.y);
    auto map_y = p.y + (sin_theta * obs.x) + (cos_theta * obs.y);
    obs.x = map_x;
    obs.y = map_y;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> resample;
  std::discrete_distribution<> distribution(weights.begin(), weights.end());
  for(int i=0; i < num_particles; i++) {
    resample.push_back(std::move(particles[distribution(gen)]));
  }
	particles = resample;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
