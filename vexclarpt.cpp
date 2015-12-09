#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <random>
#include <algorithm>
#define VEXCL_USE_CUSPARSE
#include <vexcl/constants.hpp>
#include <vexcl/devlist.hpp>
#include <vexcl/backend.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/reductor.hpp>
#include <vexcl/random.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/sort.hpp>

//todo:  command line parsing
const int num_lps = 1 << 16;
const int num_planes_per_arpt = 1;
const int num_events = num_lps * (num_planes_per_arpt + 1);
const float stop_time = 168.0f;

const int block_size = 128;
const int grid_size = ((num_events + block_size - 1) / block_size);
const int grid_run_size = ((num_lps + block_size - 1) / block_size);

const unsigned int MAX_ON_GROUND = 4;
struct AirportState {
	AirportState () : inTheAir(0), onTheGround(num_planes_per_arpt) {}
	unsigned int inTheAir; // number of aircraft landing or waiting to land
	unsigned int onTheGround; // number of landed aircraft
};

enum AirportEvents {
	ARRIVAL,
	LANDED,
	DEPARTURE,
	EMPTY,
};

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char** argv)
{
	double init_start_time = cpuSecond();

	// Get the device context.
	vex::Context ctx( vex::Filter::Count(1) && vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::DoublePrecision );
	if (!ctx) throw std::runtime_error("No devices available.");
    std::cout << ctx << std::endl;

    // RNG seed run and generator
	int random_pass = 0;
	vex::Random<float> random_state_float;
	vex::Random<unsigned int> random_state_int;

	// "Allocate" memory.
	vex::vector<unsigned int> d_event_lp_number (ctx, num_events);
	vex::vector<unsigned int> d_event_target_lp_number (ctx, num_lps);
	d_event_lp_number = (vex::element_index() % num_lps); // 0..num_lps-1

	vex::vector<float> d_event_time (ctx, num_events);
	d_event_time = random_state_float (0, (1337 << 20) + vex::element_index());

	vex::vector<unsigned int> d_event_type (ctx, num_events);
	d_event_type = (unsigned int)AirportEvents::DEPARTURE;

	vex::vector<unsigned int> d_inTheAir (ctx, num_lps);
	d_inTheAir = 0;

	vex::vector<unsigned int> d_onTheGround (ctx, num_lps);
	d_onTheGround = num_planes_per_arpt;

	vex::vector<unsigned int> d_numLandings (ctx, num_lps);
	d_numLandings = 0;

	vex::vector<unsigned int> d_events_processed (ctx, num_lps);
	d_events_processed = 0;

	vex::vector<float> d_lp_current_time (ctx, num_lps);
	d_lp_current_time = 0.0;

	vex::vector<unsigned char> d_next_event_flag (ctx, num_events);

	vex::vector<float> d_current_lbts (ctx, 1);

	float  current_lbts;

	// Additional timing variables
	double total_start_time;
	double init_duration;
	double total_duration;

	// Compile and store the simulator kernels.
	std::vector<vex::backend::kernel> initKernel;
	std::vector<vex::backend::kernel> markKernel;
	std::vector<vex::backend::kernel> markNextKernel;
	std::vector<vex::backend::kernel> simKernel;

	initKernel.emplace_back(ctx.queue(0),
		VEX_STRINGIZE_SOURCE(
				enum AirportEvents {
					ARRIVAL,
					LANDED,
					DEPARTURE,
					EMPTY,
				};

				kernel void initSim(global float *event_time,
						global unsigned int *event_types)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 16;

		const float stop_time = 168.0f;

		if (idx >=  num_lps && idx < num_lps * 2)
		{
			event_time[idx] = stop_time;
			event_types[idx] = EMPTY;
		}
	}
	),
	"initSim"
	);
	initKernel[0].config (grid_size, block_size);
	initKernel[0].push_arg(d_event_time(0));
	initKernel[0].push_arg(d_event_type(0));

	// Only need to initialize stop events once.
	initKernel[0](ctx.queue(0));
	ctx.queue(0).finish();

	markKernel.emplace_back(ctx.queue(0),
		VEX_STRINGIZE_SOURCE(
				kernel void markNextEventByLP(global unsigned int *event_lp,
						global unsigned char *next_event_flag)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 16;
		unsigned char flag;

		if (idx < num_lps * 2)
		{
			if (idx == 0 || event_lp[idx] != event_lp[idx-1])
			{
				flag = 0;
			}
			else
			{
				flag = 1;
			}
			next_event_flag[idx] = flag;
		}
	}
	),
	"markNextEventByLP"
	);
	markKernel[0].config (grid_size, block_size);
	markKernel[0].push_arg(d_event_lp_number(0));
	markKernel[0].push_arg(d_next_event_flag(0));

	simKernel.emplace_back(ctx.queue(0),
		VEX_STRINGIZE_SOURCE(
				enum AirportEvents {
					ARRIVAL,
					LANDED,
					DEPARTURE,
					EMPTY,
				};

				kernel void simulatorRun(global float *current_time,
					global float *event_time,
					global unsigned int *event_lp,
					global unsigned int *event_type,
					global unsigned int *event_lp_inTheAir,
					global unsigned int *event_lp_onTheGround,
					global unsigned int *event_lp_numLandings,
					global float *current_lbps,
					global unsigned int *target_lp,
					global unsigned int *events_processed)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 16;

		const float stop_time = 168.0f;
		const float rwy_delay_time = 0.25f;
		const float gnd_delay_time = 1.0f;
		const float look_ahead = 4.0f;

		if (idx < num_lps)
		{
			float safe_time = *current_lbps + look_ahead;

			// Check the next event
			float next_event_time = event_time[idx];
			float next_event_type = event_type[idx];

			// Ok to process?
			if(next_event_time <= safe_time &&
					next_event_time < stop_time &&
					next_event_type != EMPTY)
			{
				float cur_time = current_time[idx];
				unsigned int ev_lp = event_lp[idx];

				//sanity check
				if(cur_time > next_event_time || ev_lp != idx)
				{
					printf ("ERROR: %f > %f || %d != %d\n",
							cur_time, next_event_time, ev_lp, idx);
				}

				events_processed[idx]++;

				// Empty this event while processing it
				float new_event_time = stop_time;
				unsigned int new_event_type = EMPTY;

				if (next_event_type == ARRIVAL)
				{
					++event_lp_inTheAir[idx];
					new_event_type = LANDED;
					new_event_time = next_event_time + rwy_delay_time;
				}
				else if (next_event_type == LANDED)
				{
					++event_lp_numLandings[idx];
					--event_lp_inTheAir[idx];
					++event_lp_onTheGround[idx];
					new_event_type = DEPARTURE;
					new_event_time = next_event_time + gnd_delay_time;
				}
				else if (next_event_type == DEPARTURE)
				{
					--event_lp_onTheGround[idx];
					ev_lp = target_lp[idx];
					new_event_type = ARRIVAL;
					new_event_time = next_event_time + look_ahead;
				}

				//writes

				current_time[idx] = next_event_time;
				event_lp[idx] = ev_lp;
				event_type[idx] = new_event_type;
				event_time[idx] = new_event_time;
			}
		}
	}
	),
	"simulatorRun"
	);
	simKernel[0].config (grid_run_size, block_size);
	simKernel[0].push_arg(d_lp_current_time(0));
	simKernel[0].push_arg(d_event_time(0));
	simKernel[0].push_arg(d_event_lp_number(0));
	simKernel[0].push_arg(d_event_type(0));
	simKernel[0].push_arg(d_inTheAir(0));
	simKernel[0].push_arg(d_onTheGround(0));
	simKernel[0].push_arg(d_numLandings(0));
	simKernel[0].push_arg(d_current_lbts(0));
	simKernel[0].push_arg(d_event_target_lp_number(0));
	simKernel[0].push_arg(d_events_processed(0));

	// Initialize the Reductors.
	vex::Reductor<float, vex::MIN> min(ctx);
	vex::Reductor<unsigned int, vex::SUM> sum(ctx);

	init_duration = cpuSecond() - init_start_time;
	std::cout << "Initialization execution time for " << num_lps << " LPs was " << init_duration << " seconds" << std::endl;

	//running simulation
	std::cout << "Running simulation..." << std::endl;
	total_start_time = cpuSecond();

	while(true)
	{
		current_lbts = min(d_event_time);

		//std::cout << "Now: " << min(d_lp_current_time) /*<< d_lp_current_time*/ << "; Current LBTS: " << current_lbts << std::endl;

		d_current_lbts = current_lbts;

		if(current_lbts >= stop_time)
		{
		  break;
		}

		vex::sort_by_key (d_event_time,
				boost::fusion::vector_tie(d_event_lp_number, d_event_type),
				vex::less<double>());

		vex::sort_by_key (d_event_lp_number,
				boost::fusion::vector_tie(d_event_time, d_event_type),
				vex::less<unsigned int>());

		markKernel[0](ctx.queue(0));
		ctx.queue(0).finish();

		vex::sort_by_key (d_next_event_flag,
				boost::fusion::vector_tie(d_event_lp_number, d_event_time, d_event_type),
				vex::less<unsigned char>());

		// Select the random destinations for departures
		d_event_target_lp_number = random_state_int (0, (1337 << 20) + vex::element_index()) % num_lps;

		ctx.queue(0).finish();

		//std::cout << d_event_lp_number << std::endl;
		//std::cout << d_event_lp_number << d_event_type << d_event_time << d_inTheAir << d_onTheGround << std::endl;

		simKernel[0](ctx.queue(0));
		ctx.queue(0).finish();
	}

	total_duration = cpuSecond() - total_start_time;

	std::cout << "Stats: " << std::endl;

	unsigned int total_events_processed = sum (d_events_processed);
	unsigned int total_num_landings = sum (d_numLandings);

	std::cout << "Total Number of Events Processed: " << total_events_processed << std::endl;
	std::cout << "Total Number of Planes Landed: " << total_num_landings << std::endl;

	std::cout << "Simulation Run Time: " << total_duration << " seconds." << std::endl;

	//TODO:: Verify getting everything

	return 0;
}
