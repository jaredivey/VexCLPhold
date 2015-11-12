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
const int num_lps = 1 << 24;
const int num_events = 2 * num_lps;
const float stop_time = 240.0f;

const int block_size = 128;
const int grid_size = ((num_events + block_size - 1) / block_size);
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
	vex::Random<float> random_state_float;
	vex::Random<unsigned int> random_state_int;

	// "Allocate" memory.
	vex::vector<unsigned int> d_event_lp_number (ctx, num_events);
	vex::vector<unsigned int> d_event_target_lp_number (ctx, num_lps);
	d_event_lp_number = (vex::element_index() % num_lps); // 0..num_lps-1

	vex::vector<float> d_event_time (ctx, num_events);
	vex::vector<float> d_remote_flip (ctx, num_lps);
	d_event_time = random_state_float (0, (1337 << 20) + vex::element_index());

	vex::vector<unsigned int> d_events_processed (ctx, num_lps);
	d_events_processed = 0;
	vex::vector<float> d_lp_current_time (ctx, num_lps);
	d_lp_current_time = 0.0;

	vex::vector<unsigned char> d_next_event_flag_lp (ctx, num_events);
	vex::vector<unsigned char> d_next_event_flag_time (ctx, num_events);

	vex::vector<float> d_current_lbts (ctx, 1);

	float  current_lbts;

	// Additional timing variables
	double total_start_time;
	double init_duration;
	double total_duration;

	// Compile and store the simulator kernels.
	std::vector<vex::backend::kernel> initKernel;
	std::vector<vex::backend::kernel> markKernel;
	std::vector<vex::backend::kernel> simKernel;

	initKernel.emplace_back(ctx.queue(0),
		VEX_STRINGIZE_SOURCE(
				kernel void initStops(global float *event_time)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 24;

		const float stop_time = 240.0f;

		if (idx >= num_lps && idx < 2 * num_lps)
		{
			event_time[idx] = stop_time;
		}
	}
	),
	"initStops"
	);
	initKernel[0].config (grid_size, block_size);
	initKernel[0].push_arg(d_event_time(0));

	// Only need to initialize stop events once.
	initKernel[0](ctx.queue(0));
	ctx.queue(0).finish();

	markKernel.emplace_back(ctx.queue(0),
		VEX_STRINGIZE_SOURCE(
				kernel void markNextEventByLP(global unsigned int *event_lp,
						global unsigned char *next_event_flag_lp,
						global unsigned char *next_event_flag_time)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);

		if (idx == 0 || event_lp[idx] != event_lp[idx-1])
		{
			next_event_flag_lp[idx] = 0;
			next_event_flag_time[idx] = 0;
		}
		else
		{
			next_event_flag_lp[idx] = 1;
			next_event_flag_time[idx] = 1;
		}
	}
	),
	"markNextEventByLP"
	);
	markKernel[0].config (grid_size, block_size);
	markKernel[0].push_arg(d_event_lp_number(0));
	markKernel[0].push_arg(d_next_event_flag_lp(0));
	markKernel[0].push_arg(d_next_event_flag_time(0));

	simKernel.emplace_back(ctx.queue(0),
		VEX_STRINGIZE_SOURCE(
				kernel void simulatorRun(global float *current_time,
					global float *event_time,
					global unsigned int *event_lp,
					global float *current_lbps,
					global float *remote_flip,
					global unsigned int *target_lp,
					global unsigned int *events_processed)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 24;

		const float stop_time = 240.0f;
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
				unsigned int ev_lp = event_lp[idx];

				//sanity check
				if(cur_time > next_event_time || ev_lp != idx)
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
	simKernel[0].config (grid_run_size, block_size);
	simKernel[0].push_arg(d_lp_current_time(0));
	simKernel[0].push_arg(d_event_time(0));
	simKernel[0].push_arg(d_event_lp_number(0));
	simKernel[0].push_arg(d_current_lbts(0));
	simKernel[0].push_arg(d_remote_flip(0));
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

	double timing_loops = 0.0;
	double start_loop = 0.0;
	std::vector<double> times (13, 0.0);
	std::vector<double> durs (13, 0.0);
	while(true)
	{
		++timing_loops;
		start_loop = cpuSecond();

		current_lbts = min(d_event_time);
		times.at(0) = cpuSecond();
		durs.at(0) += times.at(0) - start_loop;

		std::cout << "Current LBTS: " << current_lbts << std::endl;
		times.at(1) = cpuSecond();
		durs.at(1) += times.at(1) - times.at(0);

		d_current_lbts = current_lbts;
		times.at(2) = cpuSecond();
		durs.at(2) += times.at(2) - times.at(1);

		if(current_lbts >= stop_time)
		{
		  break;
		}
		times.at(3) = cpuSecond();
		durs.at(3) += times.at(3) - times.at(2);

		vex::sort_by_key (d_event_time, d_event_lp_number);
		times.at(4) = cpuSecond();
		durs.at(4) += times.at(4) - times.at(3);

		vex::sort_by_key (d_event_lp_number, d_event_time);
		times.at(5) = cpuSecond();
		durs.at(5) += times.at(5) - times.at(4);

		markKernel[0](ctx.queue(0));
		ctx.queue(0).finish();
		times.at(6) = cpuSecond();
		durs.at(6) += times.at(6) - times.at(5);

		vex::sort_by_key (d_next_event_flag_lp, d_event_lp_number);
		times.at(7) = cpuSecond();
		durs.at(7) += times.at(7) - times.at(6);

		vex::sort_by_key (d_next_event_flag_time, d_event_time);
		times.at(8) = cpuSecond();
		durs.at(8) += times.at(8) - times.at(7);

		ctx.queue(0).finish();

		d_remote_flip = random_state_float (0, (1337 << 20) + vex::element_index());
		times.at(9) = cpuSecond();
		durs.at(9) += times.at(9) - times.at(8);

		d_event_target_lp_number = random_state_int (0, (1337 << 20) + vex::element_index()) % num_lps;
		times.at(10) = cpuSecond();
		durs.at(10) += times.at(10) - times.at(9);

		ctx.queue(0).finish();

		simKernel[0](ctx.queue(0));
		ctx.queue(0).finish();
		times.at(11) = cpuSecond();
		durs.at(11) += times.at(11) - times.at(10);
	}

	total_duration = cpuSecond() - total_start_time;

	std::cout << "Stats: " << std::endl;

	std::cout << "Instruction min:           " << durs.at(0) / timing_loops << std::endl;
	std::cout << "Instruction cout:          " << durs.at(1) / timing_loops << std::endl;
	std::cout << "Instruction host to dev:   " << durs.at(2) / timing_loops << std::endl;
	std::cout << "Instruction break check:   " << durs.at(3) / timing_loops << std::endl;
	std::cout << "Instruction sort by ev:    " << durs.at(4) / timing_loops << std::endl;
	std::cout << "Instruction sort by lp:    " << durs.at(5) / timing_loops << std::endl;
	std::cout << "Instruction mark kernel:   " << durs.at(6) / timing_loops << std::endl;
	std::cout << "Instruction sort next lp:  " << durs.at(7) / timing_loops << std::endl;
	std::cout << "Instruction sort next ev:  " << durs.at(8) / timing_loops << std::endl;
	std::cout << "Instruction rng rmt flip:  " << durs.at(9) / timing_loops << std::endl;
	std::cout << "Instruction rng target lp: " << durs.at(10) / timing_loops << std::endl;
	std::cout << "Instruction sim kernel:    " << durs.at(11) / timing_loops << std::endl;

	unsigned int total_events_processed = sum (d_events_processed);

	std::cout << "Total Number of Events Processed: " << total_events_processed << std::endl;

	std::cout << "Simulation Run Time: " << total_duration << " seconds." << std::endl;

	//TODO:: Verify getting everything

	return 0;
}
