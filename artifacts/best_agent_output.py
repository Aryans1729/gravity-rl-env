import math

class GravitySimulator:
    def __init__(self, G: float = 6.674e-11, dt: float = 1.0):
        """
        Initialize the gravity simulator.
        
        G: gravitational constant (m³ kg⁻¹ s⁻²)
        dt: time step in seconds
        """
        self.G = G
        self.dt = dt
        self.bodies = []  # List of body dictionaries
        self.time = 0.0
    
    def add_body(self, name: str, mass: float, x: float, y: float,
                 vx: float = 0.0, vy: float = 0.0) -> None:
        """
        Add a body to the simulation.
        Raise ValueError if this would exceed 5 bodies.
        """
        if len(self.bodies) >= 5:
            raise ValueError("Cannot add more than 5 bodies")
        
        body = {
            'name': name,
            'mass': mass,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy
        }
        self.bodies.append(body)
    
    def step(self, n: int = 1) -> None:
        """
        Advance the simulation by n steps using semi-implicit Euler integration.
        Raise ValueError if fewer than 2 bodies are present.
        Raise ZeroDivisionError if any two bodies share the same position.
        """
        if len(self.bodies) < 2:
            raise ValueError("Need at least 2 bodies to simulate")
        
        for _ in range(n):
            num_bodies = len(self.bodies)
            
            # Initialize accelerations to zero
            ax = [0.0] * num_bodies
            ay = [0.0] * num_bodies
            
            # Compute all pairwise forces and accumulate accelerations
            for i in range(num_bodies):
                for j in range(i + 1, num_bodies):
                    bi = self.bodies[i]
                    bj = self.bodies[j]
                    
                    dx = bj['x'] - bi['x']
                    dy = bj['y'] - bi['y']
                    
                    r_squared = dx * dx + dy * dy
                    
                    if r_squared == 0:
                        raise ZeroDivisionError("Two bodies at the same position")
                    
                    r = math.sqrt(r_squared)
                    
                    # Force magnitude: F = G * m_i * m_j / r²
                    F = self.G * bi['mass'] * bj['mass'] / r_squared
                    
                    # Force components on body i (towards body j)
                    Fx = F * dx / r
                    Fy = F * dy / r
                    
                    # Acceleration on body i: a = F / m
                    ax[i] += Fx / bi['mass']
                    ay[i] += Fy / bi['mass']
                    
                    # Acceleration on body j (opposite direction)
                    ax[j] -= Fx / bj['mass']
                    ay[j] -= Fy / bj['mass']
            
            # Update velocities first (semi-implicit Euler)
            for i in range(num_bodies):
                self.bodies[i]['vx'] += ax[i] * self.dt
                self.bodies[i]['vy'] += ay[i] * self.dt
            
            # Update positions using the NEW velocities
            for i in range(num_bodies):
                self.bodies[i]['x'] += self.bodies[i]['vx'] * self.dt
                self.bodies[i]['y'] += self.bodies[i]['vy'] * self.dt
            
            self.time += self.dt
    
    def get_state(self) -> list:
        """
        Return list of body dicts with keys: name, mass, x, y, vx, vy
        Bodies appear in insertion order.
        """
        # Return copies to prevent external modification
        return [
            {
                'name': b['name'],
                'mass': b['mass'],
                'x': b['x'],
                'y': b['y'],
                'vx': b['vx'],
                'vy': b['vy']
            }
            for b in self.bodies
        ]
    
    def total_energy(self) -> float:
        """
        Calculate total mechanical energy (kinetic + potential).
        KE = Σ ½ m (vx² + vy²)
        PE = Σ_{i<j} −G m_i m_j / r_ij
        """
        # Kinetic energy
        KE = 0.0
        for b in self.bodies:
            v_squared = b['vx'] ** 2 + b['vy'] ** 2
            KE += 0.5 * b['mass'] * v_squared
        
        # Potential energy
        PE = 0.0
        num_bodies = len(self.bodies)
        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                bi = self.bodies[i]
                bj = self.bodies[j]
                
                dx = bj['x'] - bi['x']
                dy = bj['y'] - bi['y']
                r = math.sqrt(dx * dx + dy * dy)
                
                if r > 0:
                    PE -= self.G * bi['mass'] * bj['mass'] / r
        
        return KE + PE
    
    def reset(self) -> None:
        """
        Clear all bodies and reset time to 0.
        """
        self.bodies = []
        self.time = 0.0
    
    def center_of_mass(self) -> tuple:
        """
        Return the mass-weighted centre of all bodies as an (x, y) tuple.
        """
        if not self.bodies:
            return (0.0, 0.0)
        
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for b in self.bodies:
            total_mass += b['mass']
            weighted_x += b['mass'] * b['x']
            weighted_y += b['mass'] * b['y']
        
        if total_mass == 0:
            return (0.0, 0.0)
        
        return (weighted_x / total_mass, weighted_y / total_mass)
    
    def orbital_period(self, name: str) -> float:
        """
        Return an estimate of the orbital period (in seconds) of the named body
        around the system's centre of mass, based on the current positions
        and masses of all bodies.
        
        For a two-body problem, the orbital period is:
        T = 2π * sqrt(a³ / (G * (M1 + M2)))
        
        where a is the semi-major axis (distance between the two bodies for circular orbit).
        
        For a body orbiting around COM, we use the reduced mass formulation.
        The body orbits at distance r from COM with effective central mass.
        
        Raise ValueError if no body with that name exists.
        """
        # Find the body
        body = None
        body_index = None
        for i, b in enumerate(self.bodies):
            if b['name'] == name:
                body = b
                body_index = i
                break
        
        if body is None:
            raise ValueError(f"No body with name '{name}' exists")
        
        # Get center of mass
        com_x, com_y = self.center_of_mass()
        
        # Distance from body to center of mass
        dx = body['x'] - com_x
        dy = body['y'] - com_y
        r = math.sqrt(dx * dx + dy * dy)
        
        # Total mass of the system
        total_mass = sum(b['mass'] for b in self.bodies)
        
        # Mass of all other bodies (central mass)
        other_mass = total_mass - body['mass']
        
        if r == 0 or other_mass == 0:
            return 0.0
        
        # For a body orbiting around COM:
        # The orbit radius around COM is r_body = r_total * M_other / M_total
        # where r_total is the separation between the bodies
        # So r_total = r_body * M_total / M_other
        
        # For a two-body system:
        # T = 2π * sqrt(a³ / (G * (M1 + M2)))
        # where a is the separation distance
        
        # The body's distance from COM is: r = a * M_other / M_total
        # So: a = r * M_total / M_other
        
        # Substituting:
        # T = 2π * sqrt((r * M_total / M_other)³ / (G * M_total))
        # T = 2π * sqrt(r³ * M_total² / (M_other³ * G))
        
        # Actually let's think more carefully:
        # T = 2π * sqrt(a³ / (G * M_total))
        # a = r * M_total / M_other
        # T = 2π * sqrt((r * M_total / M_other)³ / (G * M_total))
        # T = 2π * sqrt(r³ * M_total³ / (M_other³ * G * M_total))
        # T = 2π * sqrt(r³ * M_total² / (M_other³ * G))
        
        a = r * total_mass / other_mass
        T = 2.0 * math.pi * math.sqrt(a ** 3 / (self.G * total_mass))
        
        return T
