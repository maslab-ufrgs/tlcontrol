#!/usr/bin/env python
"""Controls all traffic lights with a q-learning algorithm.

Manages a qlearning agent for each traffic light in the network,
with an epsilon-greedy choice of action (with epsilon probability,
makes a uniformly random choice over all possible actions, otherwise
chooses the best). All usual parameters for this algorithm are
configurable via command line options.

The actions for the agent are as follows: each phase of the program
may have its duration increased or decreased, by either 5% or 20% of
its original value; also, the program may be left untouched.

To compose the observed state, the controlled lanes are grouped
according to what phase provides them with a green light (note
that each lane may be in more than one group).

The average occupancy of all lanes in each group is calculated,
and discretized according to a configurable set of intervals
that partition [0,1] (e.g. {[0,0.5], (0.5,1]}, {[0,0.2], (0.2,0.5],
(0.5,1]}).

The observed state for an agent is a tuple with the discretized
state of each group of lanes.

The reward is the mean number of halting vehicles on incoming lanes,
normalized to [-1,2] (but almost always on [-1,1].  The number of
halting vehicles should be correlated to the queue length.
"""

import optparse
import sys

import traci

from qlearning import EpsilonFirstQTable
from tlutils import Program, Lane


def function_of_iterable(fn_of_list):
    def fn_of_iterable(iterable, *args):
        if len(args) == 0:
            return fn_of_list([i for i in iterable])
        else:
            return fn_of_list([iterable] + [i for i in args])

    name = fn_of_list.__name__
    fn_of_iterable.__name__ = name
    fn_of_iterable.__doc__ = (fn_of_list.__doc__ +
                            "\n{0}(a, b, c...) = {0}([a, b, c...])".format(name))

    return fn_of_iterable

@function_of_iterable
def cartesian_prod(iterables):
    """Obtains the cartesian product of iterables.

    If ANY iterable is empty, the result will also be empty.

    Returns a list of tuples, with all possible combinations where
    the i-th element of the tuple is an element of the i-th iterable.
    """
    # prod([]) = [] and prod([a,b,c...]) = [a,b,c...]
    if len(iterables) < 2:
        return iterables

    result = [(a, b) for a in iterables[0] for b in iterables[1]]
    for i in iterables[2:]:
        result = [t + (a,) for t in result for a in i]
    return result

@function_of_iterable
def arith_mean(items):
    """Obtain the arithmetic mean of a given number of values.

    Raises a ValueError if the given iterable was empty.
    """
    if items == []:
        raise ValueError("Cannot calculate the mean of an empty iterable.")
    return float(sum(items)) / len(items)


class Agent:
    """A qlearning agent for traffic lights.

    Each phase corresponds to a 'laneset':
    the set of lanes with a green light on that phase.

    The set of states observed by the agent is the
    cartesian product of the sets of each lanesets'
    states.

    Each laneset state is the average occupation of its lanes,
    discretized according to intervals, defined by DISCRETIZATION_LIMITS.
    e.g. {[0,0.25],(0.25,0.75],(0.75,1.0]} can be obtained
    with DISCRETIZATION_LIMITS = [0.25, 0.75]

    Each phase with green lights provides an action to the
    agent: increasing its duration by 20% of its original duration,
    decreasing all other phases equally to maintain the cycle length.
    There is also a "no operation" action, which does nothing.

    The reward is the mean number of halting vehicles on incoming lanes,
    normalized to [-1,2] (but generally on [-1,1]).
    """

    DISCRETIZATION_LIMITS = [0.5]

    def __init__(self, tl_id, learning_rate, discount_factor, curiosity,
                 curiosity_decay, exploration_period, reward_expt, default_qvalue=None):
        """Initialize the agent with all parameters, and get information from SUMO."""
        self.tl_id = tl_id
        self.__lanes = tuple(traci.trafficlights.getControlledLanes(self.tl_id))
        self.__reward_expt = reward_expt

        # Obtain the current program
        programs = traci.trafficlights.getCompleteRedYellowGreenDefinition(self.tl_id)
        current_id = traci.trafficlights.getProgram(self.tl_id)
        current = next(prog for prog in programs if prog._subID == current_id)
        self.__program = Program.from_sumo(current, self.__lanes)

        # Generate the states and actions
        relevant_phases = [(idx, phase) for (idx, phase)
                           in enumerate(self.__program.phases)
                           if phase.has_green]
        num_intervals = len(self.DISCRETIZATION_LIMITS) + 1
        states = cartesian_prod(range(num_intervals) for (i, p) in relevant_phases)
        actions = [(idx, phase.duration * factor)
                   for (idx, phase) in relevant_phases
                   for factor in [0.2]] + [None]

        # Initialize the qtable
        self.__qtable = EpsilonFirstQTable(states, actions, learning_rate,
                            discount_factor, curiosity, exploration_period,
                            curiosity_decay, default_qvalue)

        # Initialize auxiliary variables
        self.__last_state = None
        self.__last_action = None

    @property
    def controlled_lanes(self):
        """Tuple of lanes controlled by this agent's trafficlight."""
        return self.__lanes

    @property
    def qvalues(self):
        """Tuple with all ((state, action), q-value) tuples in the q-table."""
        return tuple( ((st, act), self.__qtable[st, act])
                      for st  in self.__qtable.states
                      for act in self.__qtable.actions )

    def act(self, lanes):
        """Observe information from SUMO, choose an action and perform it."""
        state = self.discretize_state(lanes)

        # Update the qtable
        if self.__last_state is not None:
            reward = self.calculate_reward(lanes)
            self.__qtable.observe(self.__last_state, self.__last_action, state, reward)

        # Decide and execute the action
        action = self.__qtable.act(state)
        if action is not None:
            (inc_idx, increment) = action

            self.__program.phases[inc_idx].duration += increment

            green_phases = [(idx, p) for idx, p in enumerate(self.__program.phases)
                            if p.has_green]
            decr_phases = len(green_phases) - 1
            decrement = int(increment / decr_phases)
            for (decr_idx, phase) in green_phases:
                if decr_idx != inc_idx:
                    phase.duration -= decrement

            # Compensate errors on float->int conversion
            error = decr_phases * (increment % decr_phases)
            self.__program.phases[inc_idx].duration += error

            traci.trafficlights.setCompleteRedYellowGreenDefinition(self.tl_id,
                                                                    self.__program)

        self.__last_state = state
        self.__last_action = action

    def discretize_state(self, lanes):
        """Obtain from SUMO and discretize the occupancies of the lanes."""
        lanes_by_phase = (phase.green_lanes for phase in self.__program.phases
                          if phase.has_green)
        phase_occupancies = [arith_mean(lanes[l].mean_occupancy
                                        for l in ls)
                             for ls in lanes_by_phase]
        result = tuple(self.discretize_normalized(o) for o in phase_occupancies)
        return result


    def discretize_normalized(self, value):
        """Discretized a normalized value in [0,1] according to DISCRETIZATION_LIMITS."""
        i = 0
        for lim in self.DISCRETIZATION_LIMITS:
            if value <= lim: break
            else: i += 1
        return i

    def calculate_reward(self, lanes):
        halts = [lanes[l].normalized_mean_halting_vehicles
                 for l in self.__lanes]

        reward = 1 - arith_mean(halts)
        return (reward**self.__reward_expt * 2) - 1

# Default parameters
DEFAULT_STEP_LENGTH = 1000    # 1 second
DEFAULT_AGENT_PERIOD = 300000 # 5 minutes
DEFAULT_HARMONIC_FACTOR = 0.0
DEFAULT_REWARD_EXPT = 2
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_DISCOUNT_FACTOR = 0.7
DEFAULT_CURIOSITY = 0.2
DEFAULT_CURIOSITY_DECAY = 0.005
DEFAULT_EXPLORATION_PERIOD = 3600000 # 1 hour


def main():
    """Main body of code, implemented as a function to ease interactive debugging."""
    # Parse the options
    (options, args) = parse_options(sys.argv)
    step_length = options.step_length
    agent_period = options.agent_period

    # Connect to SUMO
    traci.init(options.port)

    # Initialize all agents
    agents = [Agent(tl_id, options.learning_rate, options.discount_factor,
                    options.curiosity, options.curiosity_decay,
                    options.exploration_period, options.reward_expt,
                    options.default_qvalue)
              for tl_id in traci.trafficlights.getIDList()]

    # Initialize all lanes
    lane_ids = set(lane for agent in agents for lane in agent.controlled_lanes)
    lanes = dict( (id, Lane(id, options.agent_period)) for id in lane_ids )

    # Simulate with control
    curr_step = 0
    while curr_step < options.end:
        # Always update the lanes
        for l in lanes.values():
            l.update()

        # Let the agents act when they're supposed to
        if curr_step % agent_period == 0:
            for a in agents:
                a.act(lanes)

        # Request another step
        traci.simulationStep(0)
        curr_step += step_length

    # Disconnect from SUMO
    traci.close()


DESCRIPTION = '\n'.join(__doc__.split('\n\n')[:2]) # First two paragraphs of docstring
EPILOG = '\n' + '\n\n'.join(__doc__.split('\n\n')[2:]) # Rest of the docstring

def parse_options(args):
    """Parse the command-line options, return (options, args)."""
    # Hack to properly print the epilog
    optparse.OptionParser.format_epilog = lambda self, formatter: self.epilog

    # Add all simulation options
    parser = optparse.OptionParser(description=DESCRIPTION,
                                   epilog=EPILOG)
    parser.add_option('-e', '--end', type="int", metavar="MILLISECONDS",
                      dest='end', help="End time of the simulation.")
    parser.add_option('-p', '--port', type="int", default=4444, dest='port',
                      help="The port where the SUMO server will be listening."
                           " [default: %default]")
    parser.add_option('--step-length', type="int", default=1000,
                      dest='step_length', metavar='MILLISECONDS',
                      help="The length of each timestep in milliseconds."
                      " [default: %default]")
    parser.add_option('-t', '--agent-period', type='int', default=DEFAULT_AGENT_PERIOD,
                      metavar="MILLISECONDS", dest='agent_period',
                      help="The number of milliseconds between each agent's action. "
                      "[default: %default]")
    parser.add_option('--reward-expt', type='float', dest='reward_expt',
                      default=DEFAULT_REWARD_EXPT, metavar='EXPT',
                      help="Exponent used for calculating the reward of an action "
                      "[default: %default]. Let Hl be the number of halting vehicles on "
                      "lane l in the last period, and Ml be the capacity of the lane l. "
                      "The reward is calculated as the sum for all lanes l of "
                      "(1 - Hl/Ml)^reward_expt.")

    # Obtain the state discretization limits
    def parse_limits(option, opt_str, value, parser):
        try:
            Agent.DISCRETIZATION_LIMITS = [float(i) for i in value.split(',')]
        except ValueError:
            parser.error("Argument for --discretization-limits must be a comma-separated list of floats.")

    parser.add_option('--discretization-limits', metavar='LIMITS',
                      action='callback', callback=parse_limits, type='string',
                      help="Limits for the intervals used on the discretization of the states. "
                      "Must be a comma-separated list of floats.")

    # Add qlearning options
    qlearning = optparse.OptionGroup(parser, "Q-Learning Parameters",
                                     "Parameters for the q-learning algorithm with "
                                     "epsilon-greedy choice.")
    qlearning.add_option('--learning-rate', type='float', dest='learning_rate', metavar='ALPHA',
                         default=DEFAULT_LEARNING_RATE, help='Same as --alpha.')
    qlearning.add_option('--alpha', type='float', dest='learning_rate', metavar='ALPHA',
                         help='The alpha parameter of q-learning. [default: %default]')

    qlearning.add_option('--discount-factor', type='float', dest='discount_factor', metavar='GAMMA',
                         default=DEFAULT_DISCOUNT_FACTOR, help='Same as --gamma.')
    qlearning.add_option('--gamma', type='float', dest='discount_factor', metavar='GAMMA',
                         help="The gamma parameter of q-learning. [default: %default]")

    qlearning.add_option('--curiosity', type='float', dest='curiosity', metavar='EPSILON',
                          default=DEFAULT_CURIOSITY, help='Same as --epsilon.')
    qlearning.add_option('--epsilon', type='float', dest='curiosity', metavar='EPSILON',
                          help='The probability of a random choice. [default: %default]')

    qlearning.add_option('--curiosity-decay', type='float',
                         dest='curiosity_decay', metavar='DECAY',
                         default=DEFAULT_CURIOSITY_DECAY, help='Same as --epsilon-decay.')
    qlearning.add_option('--epsilon-decay', type='float',
                         dest='curiosity_decay', metavar='DECAY',
                         help="The rate at which epsilon decays after every random choice "
                         "[default: %default], following: epsilon' = epsilon * (1 - epsilon_decay)")

    qlearning.add_option('--exploration-period', type='int', dest='exploration_period',
                         metavar='MILLISECONDS', default=DEFAULT_EXPLORATION_PERIOD,
                         help="The amount of time the agent is allowed to explore. "
                         "[default: %default]")

    qlearning.add_option('--default-qvalue', type='float', dest='default_qvalue',
                         metavar='QVALUE', default=None)
    parser.add_option_group(qlearning)

    # Parse the arguments
    (options, args) = parser.parse_args(args)

    # Normalize the exploration period in terms of the "action period"
    options.exploration_period /= options.agent_period

    # Verify the qlearning parameters
    try:
        EpsilonFirstQTable([], [], options.learning_rate,
                           options.discount_factor, options.curiosity,
                           options.exploration_period, options.curiosity_decay)
    except ValueError as e:
        parser.error('Invalid parameter: %s' % e)

    # Return successfuly
    return (options, args)


# Script execution
if __name__ == '__main__': main()
