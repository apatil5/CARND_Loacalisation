/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles =300;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  
  
  // Normal (Gaussian) distribution
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i=0 ; i <num_particles;i++){

    Particle temp_particle;
    temp_particle.id = i;
    temp_particle.x = dist_x(gen);
    temp_particle.y = dist_y(gen);
    temp_particle.theta = dist_theta(gen);
    temp_particle.weight = 1.0;
    particles.push_back(temp_particle);
    weights.push_back(1);
  }  

  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[],double velocity, double yaw_rate) {
    
  double v_y = (velocity/yaw_rate);
  
  double delta_theta = yaw_rate*delta_t;
  
  std::default_random_engine gen;
  

  for (int i=0 ; i <num_particles; i++){

    if(fabs(yaw_rate)<0.01){
     
     double theta_new= particles[i].theta ;
     particles[i].x +=  delta_t * velocity * cos(particles[i].theta);
     particles[i].y +=  delta_t * velocity * sin(particles[i].theta);
     particles[i].theta =theta_new ;}
  
    else{
     double theta_new= particles[i].theta + delta_theta;
     particles[i].x += v_y*(sin(theta_new) - sin(particles[i].theta)) ;
     particles[i].y += -1* v_y*(cos(theta_new) - cos(particles[i].theta)) ;
     particles[i].theta =theta_new ;
    }
 
    //adding sensor noise to predictions
    std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);  

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);  
  
  }
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
        
        for (int i = 0; i < observations.size(); i++) {
       
          double d = 50 ;
          for (int j = 0; j < predicted.size(); j++) {
              
            double temp = dist( observations[i].x, observations[i].y, predicted[j].x,predicted[j].y);

            if (temp < d) {
              d = temp;
	      observations[i].id = predicted[j].id;
	    }
          }
        }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs> &observations, const Map &map_landmarks) {
 
    //const double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    //This term can be excluded from the calculation because this term cancels out when weights are normalized
  
    double ws=0;
    
    for (int j=0 ; j<num_particles ; j++){
    
     //List of observed locations
     
     vector<LandmarkObs> TOBS(observations.size());
    
     for (int i=0;i<observations.size();i++){
   
       LandmarkObs transformed_obs;
       TOBS[i].id = i;
       TOBS[i].x = particles[j].x + observations[i].x*cos(particles[j].theta)-sin(particles[j].theta)*observations[i].y ; 
       TOBS[i].y = particles[j].y + observations[i].y*cos(particles[j].theta)+sin(particles[j].theta)*observations[i].x ;
     }
    
   //List of Map locations with in 50 m range
     
     vector<LandmarkObs> POBS;
     LandmarkObs temp_POB; 
     int z=0;
     for (int b =0;b<map_landmarks.landmark_list.size();b++){
       
       double temp=dist(map_landmarks.landmark_list[b].x_f,map_landmarks.landmark_list[b].y_f,particles[j].x,particles[j].y);
       if (temp < sensor_range){
      	temp_POB.x  = map_landmarks.landmark_list[b].x_f;
	temp_POB.y  = map_landmarks.landmark_list[b].y_f;
	temp_POB.id = b; 
        POBS.push_back(temp_POB);
       }
       
     }
     
     //Couplped observation indices
     dataAssociation(POBS,TOBS);// output is TOBS has POBS closest lanmark id

     double prob = 1;
     double x2= 2.0 * pow(std_landmark[0],2);
     double y2= 2.0 * pow(std_landmark[0],2);

     for (int k=0 ;k < TOBS.size(); k++){
      
       double x_obs = TOBS[k].x;
       double mu_x  = map_landmarks.landmark_list[TOBS[k].id].x_f;
       double y_obs = TOBS[k].y;
       double mu_y  = map_landmarks.landmark_list[TOBS[k].id].y_f;
       
       prob*= exp(-1.0 * (((pow((x_obs - mu_x), 2)/(x2))) + (pow((y_obs - mu_y), 2)/(y2))));
       
     }
     
     weights[j]=prob; 
     ws+=prob;
    }

     for(int i=0 ; i < weights.size() ; i++){
       weights[i]=weights[i]/ws;//Normalized weights
     
     }
    
}


void ParticleFilter::resample() {
  
        // Random particle index and beta value
        default_random_engine gen;

        //Generate random particle index
        uniform_int_distribution<int> p_idx(0, num_particles - 1);

        int index = p_idx(gen);

        double beta = 0.0;

        double mw = 2.0 * *max_element(weights.begin(), weights.end());

        for (int i = 0; i < particles.size(); i++) {

          uniform_real_distribution<double> random_weight(0.0, mw);
          beta += random_weight(gen);

          while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
          }
          
	    particles[i] = particles[index];
        }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

  
