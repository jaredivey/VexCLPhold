#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <random>
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
const int num_lps = 1 << 20;
const float stop_time = 60.0f;

const int block_size = 128;
const int grid_run_size = ((num_lps + block_size - 1) / block_size);

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
	vex::Context ctx( vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::DoublePrecision );
	if (!ctx) throw std::runtime_error("No devices available.");
    std::cout << ctx << std::endl;

    // RNG seed run and generator
	int random_pass = 0;
	vex::Random<float> random_state;

	// "Allocate" memory.
	vex::vector<int> d_event_lp_number (ctx, num_lps);
	vex::vector<int> d_event_target_lp_number (ctx, num_lps);
	d_event_lp_number = vex::element_index(); // 0..num_lps-1

	vex::vector<float> d_event_time (ctx, num_lps);
	vex::vector<float> d_remote_flip (ctx, num_lps);
	d_event_time = random_state (vex::element_index(), random_pass++);

	vex::vector<unsigned int> d_events_processed (ctx, num_lps);
	d_events_processed = 0;
	vex::vector<float> d_lp_current_time (ctx, num_lps);
	d_lp_current_time = 0.0;

	vex::vector<float> d_current_lbts (ctx, 1);

	float  current_lbts;

	// Additional timing variables
	double total_start_time;
	double init_duration;
	double total_duration;

	// Compile and store the simulator kernel.
	std::vector<vex::backend::kernel> simKernel;
	for(uint d = 0; d < ctx.size(); d++)
	{
	    simKernel.emplace_back(ctx.queue(d),
	        VEX_STRINGIZE_SOURCE(
	        		kernel void simulatorRun(global float *current_time,
	            		global float *event_time,
	            		global int *event_lp,
	            		global float *current_lbps,
	            		global float *remote_flip,
	            		global float *target_lp,
	            		global int *events_processed)
	    {
	    	const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
	    	//printf ("ID: %d, local: %d; group: %d\n", idx, get_local_id (0), get_group_id (0));
	    	const int num_lps = 1 << 20;

	    	const float stop_time = 60.0f;
	    	const float local_rate = 0.9f;
	    	const float delay_time = 0.9f;
	    	const float look_ahead = 4.0f;

	    	if (idx < num_lps)
	    	{
				float safe_time = *current_lbps + look_ahead;

				// Check the next event
				float next_event_time = event_time[idx];

				// Ok to process?
				if(next_event_time <= safe_time && next_event_time < stop_time)
				{
					float cur_time = current_time[idx];
					int ev_lp = event_lp[idx];

					//sanity check
					if(cur_time > next_event_time)// || ev_lp != idx)
					{
						printf ("ERROR\n");
					}

					events_processed[idx]++;

					//next_event_time stores current time if we reach here
					float new_event_time = delay_time + next_event_time;

					if(remote_flip[idx] < local_rate)
					{
						event_lp[idx] = idx;
					}
					else
					{
						//target_lp could be me, however we'll assume that the probability is small.
						event_lp[idx] = target_lp[idx];
						new_event_time += look_ahead;
					}

					//writes

					current_time[idx] = next_event_time;
					event_time[idx] = new_event_time;
				}
	    	}
	    }
	    ),
	    "simulatorRun"
	    );
	}

	// Initialize the Reductors.
	vex::Reductor<float, vex::MIN> min(ctx);
	vex::Reductor<int, vex::SUM> sum(ctx);

	init_duration = cpuSecond() - init_start_time;
	std::cout << "Initialization execution time for " << num_lps << " LPs was " << init_duration << " seconds" << std::endl;

	//running simulation
	std::cout << "Running simulation..." << std::endl;
	total_start_time = cpuSecond();
	while(true)
	{
		current_lbts = min(d_event_time);
		d_current_lbts = min(d_event_time);
		std::cout << "Current LBTS: " << current_lbts << std::endl;

		if(current_lbts >= stop_time)
		{
		  break;
		}

		vex::sort_by_key (d_event_time, d_event_lp_number);
		vex::sort_by_key (d_event_lp_number, d_event_time);

		d_remote_flip = random_state (vex::element_index(), random_pass++);
		d_event_target_lp_number = (num_lps - 1) * random_state (vex::element_index(), random_pass++);

		for(uint d = 0; d < ctx.size(); d++) {
			simKernel[d].config (grid_run_size, block_size);
			simKernel[d].push_arg(d_lp_current_time(d));
			simKernel[d].push_arg(d_event_time(d));
			simKernel[d].push_arg(d_event_lp_number(d));
			simKernel[d].push_arg(d_current_lbts(d));
			simKernel[d].push_arg(d_remote_flip(d));
			simKernel[d].push_arg(d_event_target_lp_number(d));
			simKernel[d].push_arg(d_events_processed(d));
			simKernel[d](ctx.queue(d));
			ctx.queue(d).finish();
		}
	}

	total_duration = cpuSecond() - total_start_time;

	std::cout << "Stats: " << std::endl;

	unsigned int total_events_processed = sum (d_events_processed);

	std::cout << "Total Number of Events Processed: " << total_events_processed << std::endl;

	std::cout << "Simulation Run Time: " << total_duration << " seconds." << std::endl;

	//TODO:: Verify getting everything

	return 0;
}