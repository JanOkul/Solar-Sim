import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Runs and controls the simulation.
class Simulation:

    def __init__(self, satellite=False):
        # Opens json file and retrieves data.
        with open("input_data.json", 'r') as file:
            data = json.load(file)

        # This opens and closes the energy.txt file as a write file which clears the file.
        open("energy.txt", 'w').close()

        # Assign simulation data.
        sim_data = data["simulation"]
        self.TIME_STEP = 10000
        self.STEPS = sim_data["no_of_steps"]
        self.G = sim_data["G"]
        self.time = 0
        self.satellite = satellite

        # The system's star is created as a Body object
        self.star = Body(*data["star"], self.TIME_STEP, self.G)

        # Each planet is created as a Body object.
        self.number_of_planets = len(data["planets"])
        self.planets = []

        # Adds each planet in json a list as an object.
        for planet in data["planets"]:
            self.planets.append(Body(*planet, self.TIME_STEP, self.G, self.star.MASS))

        self.bodies = [self.star] + self.planets    # Bodies is a list of all bodies mainly for acceleration calculation.
        self.number_of_bodies = len(self.bodies)
        self.display_bodies = []  # This variable stores all the matplotlib objects for all the objects as a list.

        # Variables for calculating the average energy per day.
        self.energy = 0
        self.iterations = 0
        self.days_first_occurence = [True, 0]

        # Creates a satellite if the parameter has been entered as true.
        if self.satellite:
            self.bodies.append(self.create_satellite(data["satellite"]))
            self.number_of_bodies = len(self.bodies)

    # Calculates acceleration for all the bodies.
    def calculate_acceleration(self, target_body):
        acceleration = 0

        # Mass and position of the current body.
        target_index = [body.NAME for body in self.bodies].index(target_body)
        mass_i, position_i = self.bodies[target_index].MASS, self.bodies[target_index].position

        # Mass and position for the other bodies is initialised.
        mass_j, position_j = [], []

        # Gets mass and position of every body that is not the target body.
        for i in range(self.number_of_bodies):
            if i != target_index:

                # Split mass and position tuple into separate variables.
                mass_j.append(self.bodies[i].MASS)
                position_j.append(self.bodies[i].position)

        # Goes through every body and sums up gravitation.
        for i in range((self.number_of_bodies - 1)):
            position_ji = position_j[i] - position_i
            acceleration += (mass_j[i] / (np.linalg.norm(position_ji) ** 3)) * position_ji

        acceleration = self.G * acceleration
        # Returns acceleration to bodies object.
        self.bodies[target_index].acceleration[2] = acceleration

    # Calculates the potential energy of the system at the current time step.
    def calculate_pe(self):
        total_pe = 0

        # Calculates potential energy for every single body in the system.
        for i in range(self.number_of_bodies):
            mass_i, position_i = self.bodies[i].MASS, self.bodies[i].position

            # Calculates potential energy for between body i and every other body j.
            for j in range(self.number_of_bodies):
                mass_j, position_j = self.bodies[j].MASS, self.bodies[j].position

                # Makes sure body i is not being compared to itself in j.
                if i != j:
                    total_pe += (self.G * mass_i * mass_j) / np.linalg.norm(position_i - position_j)

            total_pe = -0.5 * total_pe

        return total_pe

    # Calculates total kinetic energy in the system.
    def calculate_total_ke(self):
        total_ke = 0

        # Sums up total kinetic energy in all the bodies.
        for body in self.bodies:
            total_ke += body.calculate_ke()

        return total_ke

    # Calculates the sum of the kinetic and potential energy in the system.
    def calculate_total_energy(self):
        return self.calculate_pe() + self.calculate_total_ke()

    # Controls the flow of the simulation.
    def simulate(self, i):
        self.time += self.TIME_STEP
        self.check_if_new_day()

        # Sends data to the probe sensor to check how close to Mars it is.
        if self.satellite:
            index = [body.NAME for body in self.bodies].index("Mars")  # Index of launch planet.
            self.bodies[-1].sensors(self.bodies[index].position, self.time)

        # Updates position of all bodies.
        for body in self.bodies:
            body.update_position()

        # Updates acceleration of all bodies.
        for i in range(self.number_of_bodies):
            self.calculate_acceleration(self.bodies[i].NAME)

        # Updates velocity of all bodies.
        for body in self.bodies:
            body.update_velocity()

        # Repositions all plt objects and checks for orbital period.
        for i in range(self.number_of_bodies):
            position = self.bodies[i].position
            self.display_bodies[i].center = position
            self.bodies[i].print_orbital_period(self.time)

        return self.display_bodies  # This list will be used by FuncAnimation.

    # Displays the simulation.
    def run_simulation(self):
        fig, ax = plt.figure(), plt.axes()  # Create figure and axes.

        # Create all the bodies.
        for body in self.bodies:
            self.display_bodies.append(body.body)

        # Add bodies to the axes.
        for body in self.bodies:
            ax.add_patch(body.body)

        ax.axis("scaled")   # Sets limits for visualisation.

        excess = 1.1  # Excess adds some extra space around the bodies, so they don't go off the screen.
        base_limit = max([body.position[0] for body in self.bodies])
        total_limit = base_limit * excess

        # Sets axes limit.
        ax.set_xlim(-total_limit, total_limit)
        ax.set_ylim(-total_limit, total_limit)

        # Creates and runs animation by repeatedly calling simulate.
        animation = FuncAnimation(fig, self.simulate, self.STEPS, repeat=False, interval=1, blit=True)
        plt.title("Simulation of solar system")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.legend(list(map(lambda x: x.NAME, self.bodies)), loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

    # Calculates the average energy each day and then calls method to write to file.
    def check_if_new_day(self):
        seconds_to_days = 60 * 60 * 24      # Convert seconds to earth days.
        day = self.time / seconds_to_days   # Current day and decimal value earth through day

        # Checks if this is the first occurence of that day.
        if int(day) > self.days_first_occurence[1]:
            self.days_first_occurence = [True, int(day)]
        else:
            self.days_first_occurence[0] = False

        # If this is the first occurence of that day then that means a new day has started so send the average energy
        # last day to file and reset variables for next day.
        if self.days_first_occurence[0]:
            self.write_energy_to_file(int(day), self.energy/self.iterations)
            self.energy = 0
            self.iterations = 0
        # Calculate total energy and amount of energies added on.
        else:
            self.energy += self.calculate_total_energy()
            self.iterations += 1

    # Writes total energy to a text file.
    @staticmethod
    def write_energy_to_file(day, energy):
        with open("energy.txt", 'a') as write:
            write.write(str(int(day)) + "," + str(energy) + "\n")

    # Prints a graph of the total energy of the system per day that the simulation ran for.
    @staticmethod
    def graph_energy():
        day, energy = [], []        # X and Y axis values for graph

        # Read each line of energy file and append it to the day and energy lists.
        with open("energy.txt", 'r') as file:
            while line := file.readline():       # := assigns and compared simultaneously.
                line = line.replace('\n', '').split(',')        # Format string to string list of day and energy strings.
                line = [float(line[i]) for i in range(len(line))]   # Convert string list to a list of floats

                # Appends line to both variables.
                day.append(line[0])
                energy.append(line[1])

        # Plot and show energy graph.
        plt.plot(day,energy)
        plt.title("Total Energy in System")
        plt.xlabel("Earth Days")
        plt.ylabel("Energy (J)")
        plt.show()

    def create_satellite(self, data):
        index = [body.NAME for body in self.bodies].index("Earth")      # Index of launch planet.
        mass = self.bodies[index].MASS                                  # Mass of launch planet.

        return Satellite(*data["properties"], self.TIME_STEP, self.G, mass, data["launch_velocity"])


# Holds properties and methods that each body has and not related to the simulation.
class Body:

    def __init__(self, NAME, COLOUR, MASS, POSITION, SIZE, TIME_STEP, G, star_mass=0):
        # Assigns bodies constants.
        self.NAME = NAME
        self.COLOUR = COLOUR
        self.MASS = MASS
        self.TIME_STEP = TIME_STEP

        # Variables to do with calculating position of body.
        self.position = np.array(POSITION)
        self.acceleration = [np.zeros(2)] * 3

        # Variables to do with calculating the orbital period of a planets.
        self.last_y_negative = True
        self.this_y_negative = True
        self.sum_orbital_periods = 0
        self.last_orbital_period = 0
        self.orbits = 0

        # Other variables
        self.body = plt.Circle((self.position[0], self.position[1]), SIZE, color=self.COLOUR, animated=True)

        # Makes sure that the sun or any central body does not create a division by zero as position could be 0
        try:
            v = np.sqrt(G * star_mass / float(self.position[0]))
            self.velocity = np.array([0, v])
        except ZeroDivisionError:
            self.velocity = np.zeros(2)

    # Updates body's position.
    def update_position(self):
        self.position = self.position + (self.velocity * self.TIME_STEP) + (1 / 6) * (4 * self.acceleration[1] - self.acceleration[0]) * (self.TIME_STEP ** 2)
        self.body.center = (self.position[0], self.position[1])

    # Updates body's velocity and shift accelerations for next time step.
    def update_velocity(self):
        self.velocity = self.velocity + (1 / 6) * (2 * self.acceleration[2] + (5 * self.acceleration[1]) - self.acceleration[0]) * self.TIME_STEP
        self.shift_acceleration()

    # Shifts all accelerations one place to the left as accelerations become further in the past as the time increases.
    def shift_acceleration(self):
        self.acceleration[0] = self.acceleration[1]
        self.acceleration[1] = self.acceleration[2]

    # Calculates kinetic energy of body.
    def calculate_ke(self):
        ke = 0.5 * self.MASS * np.linalg.norm(self.velocity ** 2)
        return ke

    # Prints orbital period for body
    def print_orbital_period(self, time):
        self.last_y_negative = self.this_y_negative
        seconds_to_years = 60 * 60 * 24 * 365

        # Checks whether current position is in the positive y-axis.
        if self.position[1] >= 0:
            self.this_y_negative = False
        else:
            self.this_y_negative = True

        # Checks if y-axis of body just went from being in negative to being in positive meaning it has done an orbit.
        if (self.last_y_negative is True) and (self.this_y_negative is False) and (time > self.TIME_STEP):
            self.orbits += 1
            self.sum_orbital_periods += self.last_orbital_period
            self.last_orbital_period = time - self.sum_orbital_periods

            print(f"Orbit number: {self.orbits} of {self.NAME}, Orbit time: {round(self.last_orbital_period/seconds_to_years, 5)} Earth years.")


# This class is a subclass of Body as the satellite needs special launch velocities.
class Satellite(Body):
    def __init__(self, NAME, COLOUR, MASS, POSITION, SIZE, TIME_STEP, G, star_mass=0, launch_velocity=0):
        Body.__init__(self, NAME, COLOUR, MASS, POSITION, SIZE, TIME_STEP, G, star_mass=star_mass)
        self.velocity = np.array(launch_velocity)
        self.closest_distance = self.position
        self.print_distance = -1        # Controls whether the closest distance is printed or not.

    # Override method as satellite does not need orbital period.
    def print_orbital_period(self, time):
        pass

    # Prints satellite statistics
    def sensors(self, position, time):
        position_difference = position - self.position      # Distance between Mars and Earth.

        # If the new distance is less than the last then set the closest distance as the new one
        if np.linalg.norm(self.closest_distance) > np.linalg.norm(position_difference):
            self.closest_distance = position_difference
            self.print_distance = -1
        else:
            self.print_distance += 1

        # Only prints out when the closest distance hasn't been reached.
        if self.print_distance == 0:
            print(f"Closest distance to planet was {int(np.linalg.norm(self.closest_distance)/1000)}km, at {int(time/(60*60*24))} days")


# Initialises simulation and runs the simulation.
def main():
    satellite = True if sys.argv[1] == "--satellite" else False

    SIMULATION = Simulation(satellite)
    SIMULATION.run_simulation()
    SIMULATION.graph_energy()

if __name__ == "__main__":
    main()
