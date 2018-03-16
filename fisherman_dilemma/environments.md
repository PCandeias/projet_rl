## Random fisherman environment:
- A population of $N$ fisherman can take fish from the pond
- There are $K$ independant agents, $N-K$ preprogrammed agents
- The preprogrammed agents take a fish from the pond with probably $p_f$
- Stock increases $X$ times in size at each time step, with a max stock size of $Y$
- There are $S$ steps per episode

Things to play with:
- Change the values of the different parameters
- Add a varying ammounts of indenpendant agents

## Coordinated fisherman environment:
- A population of $N$ fisherman can take fish from the pond
- There are $K$ independant agents, $N-K$ preprogrammed agents
- There is a signal that takes values from ${0,1,...,X-1}$
- The preprogrammed agents are divided into X groups
- Stock increases $X$ times in size at each time step, with a max stock size of $Y$
- There are $S$ steps per episode

Things to play with:
- Change the values of the different parameters
- Add a varying ammounts of indenpendant agents
- There could be a seperate signal for each agent assigning it to a group, or could leave the agent to guess which groups to join
