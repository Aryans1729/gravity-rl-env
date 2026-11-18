# Correct, but written differently: pairwise forces with Newton's 3rd law, r^3 form.
import math
class GravitySimulator:
    def __init__(self, G=6.674e-11, dt=1.0):
        self.G, self.dt, self.b = G, dt, []
    def add_body(self, name, mass, x, y, vx=0.0, vy=0.0):
        if len(self.b) == 5: raise ValueError("max 5")
        self.b.append([name, mass, x, y, vx, vy])
    def reset(self): self.b = []
    def step(self, n=1):
        if len(self.b) < 2: raise ValueError("need 2")
        for _ in range(n):
            fx = [0.0]*len(self.b); fy = [0.0]*len(self.b)
            for i in range(len(self.b)):
                for j in range(i+1, len(self.b)):
                    dx = self.b[j][2]-self.b[i][2]; dy = self.b[j][3]-self.b[i][3]
                    r3 = (dx*dx+dy*dy)**1.5
                    f = self.G*self.b[i][1]*self.b[j][1]/r3
                    fx[i] += f*dx; fy[i] += f*dy; fx[j] -= f*dx; fy[j] -= f*dy
            for k, bb in enumerate(self.b):
                bb[4] += fx[k]/bb[1]*self.dt; bb[5] += fy[k]/bb[1]*self.dt
            for bb in self.b:
                bb[2] += bb[4]*self.dt; bb[3] += bb[5]*self.dt
    def get_state(self):
        return [dict(name=n, mass=m, x=x, y=y, vx=vx, vy=vy) for n, m, x, y, vx, vy in self.b]
    def total_energy(self):
        e = sum(0.5*m*(vx*vx+vy*vy) for _, m, _, _, vx, vy in self.b)
        for i in range(len(self.b)):
            for j in range(i+1, len(self.b)):
                e -= self.G*self.b[i][1]*self.b[j][1]/math.dist(self.b[i][2:4], self.b[j][2:4])
        return e
    def center_of_mass(self):
        M = sum(b[1] for b in self.b)
        return (sum(b[1]*b[2] for b in self.b)/M, sum(b[1]*b[3] for b in self.b)/M)
    def orbital_period(self, name):
        (_, m1, x1, y1, a1, b1), (_, m2, x2, y2, a2, b2) = self.b
        mu = self.G*(m1+m2); r = math.dist((x1, y1), (x2, y2)); v2 = (a2-a1)**2+(b2-b1)**2
        a = 1/(2/r - v2/mu)
        return 2*math.pi*math.sqrt(a**3/mu)
