#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <numeric>
#include <random>
#include <algorithm>
#include <sys/time.h>

// Boost.Compute headers
#include <compute/boost/compute/core.hpp>
#include <compute/boost/compute/utility/source.hpp>
#include <compute/boost/compute/function.hpp>
#include <compute/boost/compute/algorithm/iota.hpp>
#include <compute/boost/compute/algorithm/none_of.hpp>
#include <compute/boost/compute/algorithm/transform.hpp>
#include <compute/boost/compute/algorithm/unique.hpp>
#include <compute/boost/compute/container/vector.hpp>
#include <compute/boost/compute/random/default_random_engine.hpp>
#include <compute/boost/compute/random/uniform_real_distribution.hpp>
#include <compute/boost/compute/random/uniform_int_distribution.hpp>
#include <compute/boost/compute/algorithm/fill.hpp>
#include <compute/boost/compute/algorithm/min_element.hpp>
#include <compute/boost/compute/algorithm/reduce.hpp>
#include <compute/boost/compute/algorithm/sort_by_key.hpp>
#include <compute/boost/compute/detail/print_range.hpp>

namespace compute = boost::compute;
using compute::uint_;

//todo:  command line parsing
const uint_ num_lps = 1 << 12;
const uint_ num_events = 2 * num_lps;
const float stop_time = 60.0f;

const uint_ block_size = 1024;
const uint_ grid_size = ((num_events + block_size - 1) / block_size);
const uint_ grid_run_size = ((num_lps + block_size - 1) / block_size);

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
    compute::device gpu = compute::system::default_device();

    // create a compute context and command queue
    compute::context ctx(gpu);
    compute::command_queue queue(ctx, gpu);
	if (!ctx) throw std::runtime_error("No devices available.");
    std::cout << ctx << std::endl;
    std::cout << "\tDevice name: " << gpu.name () << std::endl;
    std::cout << "\tDevice type: " << gpu.type () << std::endl;
    std::cout << "\tDevice vendor: " << gpu.vendor () << std::endl;
    std::cout << "\tDevice profile: " << gpu.profile () << std::endl;
    std::cout << "\tDevice version: " << gpu.version () << std::endl;
    std::cout << "\tDevice driver version: " << gpu.driver_version () << std::endl;
    for (unsigned int i = 0; i < gpu.extensions ().size(); ++i)
    	std::cout << "\tDevice extensions: " << gpu.extensions ().at (i) << std::endl;
    std::cout << "\tDevice address bits: " << gpu.address_bits () << std::endl;
    std::cout << "\tDevice global memory size: " << gpu.global_memory_size () << std::endl;
    std::cout << "\tDevice local memory size: " << gpu.local_memory_size () << std::endl;
    std::cout << "\tDevice clock frequency: " << gpu.clock_frequency () << std::endl;
    std::cout << "\tDevice compute units: " << gpu.compute_units () << std::endl;
    std::cout << "\tDevice max memory alloc size: " << gpu.max_memory_alloc_size () << std::endl;
    std::cout << "\tDevice max work group size: " << gpu.max_work_group_size () << std::endl;
    std::cout << "\tDevice max work item dimensions: " << gpu.max_work_item_dimensions () << std::endl;
    std::cout << "\tDevice profiling timer resolution: " << gpu.profiling_timer_resolution () << std::endl;

    // RNG seed run and generator
	compute::default_random_engine random_engine(queue);
	compute::uniform_real_distribution<float> random_state_float(0.f, 1.f);
	compute::uniform_int_distribution<uint_> random_state_int(0, num_lps - 1);

	// "Allocate" memory.
	compute::vector<uint_> d_event_lp_number(num_events, ctx);
	compute::vector<uint_> d_event_target_lp_number(num_lps, ctx);

	BOOST_COMPUTE_FUNCTION(uint_, init_lps, (uint_ x),
		    {
		       return (x % (1 << 12));
		    });

	compute::iota(d_event_lp_number.begin(), d_event_lp_number.end(), 0, queue);
	compute::transform(d_event_lp_number.begin(), d_event_lp_number.end(),
			d_event_lp_number.begin(), init_lps, queue); // 0..num_lps-1

	compute::vector<float> d_event_time(num_events, ctx);
	compute::vector<float> d_remote_flip(num_lps, ctx);
	random_state_float.generate (d_event_time.begin(), d_event_time.end(), random_engine, queue);

	compute::vector<uint_> d_events_processed(num_lps, ctx);
	compute::fill (d_events_processed.begin(), d_events_processed.end(), 0, queue);
	compute::vector<float> d_lp_current_time(num_lps, ctx);
	compute::fill (d_lp_current_time.begin(), d_lp_current_time.end(), 0.0, queue);

	compute::vector<float> d_current_lbts(1, ctx);
	float  current_lbts;

	compute::vector<unsigned char> d_next_event_flag_lp (num_events, ctx);
	compute::vector<unsigned char> d_next_event_flag_time (num_events, ctx);

	// Additional timing variables
	double total_start_time;
	double init_duration;
	double total_duration;

	// Compile and store the simulator kernels.
	std::stringstream options;
	options << "-DTILE_DIM=" << 1 << " -DBLOCK_ROWS=" << block_size;

	const char source_initStops[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
				kernel void initStops(global float *event_time)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 12;

		const float stop_time = 60.0f;

		if (idx >= num_lps && idx < 2 * num_lps)
		{
			event_time[idx] = stop_time;
		}
	}
	);
    compute::program program_initStops =
        compute::program::build_with_source(source_initStops, ctx, options.str());

    compute::kernel initKernel = program_initStops.create_kernel("initStops");
    initKernel.set_arg(0, d_event_time);

    compute::event start_initStops;
    start_initStops = queue.enqueue_1d_range_kernel(initKernel, 0, grid_size*block_size, block_size);
    queue.finish();

	const char source_markLPs[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
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
	);

    compute::program program_markLPs =
        compute::program::build_with_source(source_markLPs, ctx, options.str());

    compute::kernel markKernel = program_markLPs.create_kernel("markNextEventByLP");
    markKernel.set_arg(0, d_event_lp_number);
    markKernel.set_arg(1, d_next_event_flag_lp);
    markKernel.set_arg(2, d_next_event_flag_time);

	const char source_simRun[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
				kernel void simulatorRun(global float *current_time,
					global float *event_time,
					global unsigned int *event_lp,
					global float *current_lbps,
					global float *remote_flip,
					global unsigned int *target_lp,
					global unsigned int *events_processed)
	{
		const size_t idx = get_local_id(0) + get_group_id(0) * get_local_size(0);
		const unsigned int num_lps = 1 << 12;

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
	);

    compute::program program_simRun =
        compute::program::build_with_source(source_simRun, ctx, options.str());

    compute::kernel simKernel = program_simRun.create_kernel("simulatorRun");
    simKernel.set_arg(0, d_lp_current_time);
    simKernel.set_arg(1, d_event_time);
    simKernel.set_arg(2, d_event_lp_number);
    simKernel.set_arg(3, d_current_lbts);
    simKernel.set_arg(4, d_remote_flip);
    simKernel.set_arg(5, d_event_target_lp_number);
    simKernel.set_arg(6, d_events_processed);

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

		current_lbts = *compute::min_element(d_event_time.begin(), d_event_time.end(), queue);
		times.at(0) = cpuSecond();
		durs.at(0) += times.at(0) - start_loop;

		std::cout << "Current LBTS: " << current_lbts << std::endl;
		times.at(1) = cpuSecond();
		durs.at(1) += times.at(1) - times.at(0);

		compute::fill (d_current_lbts.begin(), d_current_lbts.end(), current_lbts, queue);
		times.at(2) = cpuSecond();
		durs.at(2) += times.at(2) - times.at(1);

		if(current_lbts >= stop_time)
		{
		  break;
		}
		times.at(3) = cpuSecond();
		durs.at(3) += times.at(3) - times.at(2);

		compute::sort_by_key(d_event_time.begin(), d_event_time.end(), d_event_lp_number.begin(), queue);
		times.at(4) = cpuSecond();
		durs.at(4) += times.at(4) - times.at(3);

		compute::sort_by_key(d_event_lp_number.begin(), d_event_lp_number.end(), d_event_time.begin(), queue);
		times.at(5) = cpuSecond();
		durs.at(5) += times.at(5) - times.at(4);

		compute::event start_markLPs;
	    markKernel.set_arg(0, d_event_lp_number);
	    markKernel.set_arg(1, d_next_event_flag_lp);
	    markKernel.set_arg(2, d_next_event_flag_time);
		start_markLPs = queue.enqueue_1d_range_kernel(markKernel, 0, grid_size*block_size, block_size);
	    queue.finish();
		times.at(6) = cpuSecond();
		durs.at(6) += times.at(6) - times.at(5);

		compute::sort_by_key(d_next_event_flag_lp.begin(), d_next_event_flag_lp.end(), d_event_lp_number.begin(), queue);
		times.at(7) = cpuSecond();
		durs.at(7) += times.at(7) - times.at(6);

		compute::sort_by_key(d_next_event_flag_time.begin(), d_next_event_flag_time.end(), d_event_time.begin(), queue);
		times.at(8) = cpuSecond();
		durs.at(8) += times.at(8) - times.at(7);

		queue.finish();

		random_state_float.generate (d_remote_flip.begin(), d_remote_flip.end(), random_engine, queue);
		times.at(9) = cpuSecond();
		durs.at(9) += times.at(9) - times.at(8);

		random_state_int.generate (d_event_target_lp_number.begin(), d_event_target_lp_number.end(), random_engine, queue);
		times.at(10) = cpuSecond();
		durs.at(10) += times.at(10) - times.at(9);

		compute::event start_simRun;
	    simKernel.set_arg(0, d_lp_current_time);
	    simKernel.set_arg(1, d_event_time);
	    simKernel.set_arg(2, d_event_lp_number);
	    simKernel.set_arg(3, d_current_lbts);
	    simKernel.set_arg(4, d_remote_flip);
	    simKernel.set_arg(5, d_event_target_lp_number);
	    simKernel.set_arg(6, d_events_processed);
		start_simRun = queue.enqueue_1d_range_kernel(simKernel, 0, grid_run_size*block_size, block_size);
	    queue.finish();
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

	unsigned int total_events_processed = 0;
	compute::reduce(d_events_processed.begin(), d_events_processed.end(), &total_events_processed, compute::plus<int>(), queue);

	std::cout << "Total Number of Events Processed: " << total_events_processed << std::endl;

	std::cout << "Simulation Run Time: " << total_duration << " seconds." << std::endl;

	//TODO:: Verify getting everything

	return 0;
}
