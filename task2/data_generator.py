import random
import sys

def generate(n):
    filename = f"data_{n}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for i in range(n):
            mass = random.uniform(1e9, 1e12)
            x = random.uniform(-100.0, 100.0)
            y = random.uniform(-100.0, 100.0)
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(-2.0, 2.0)
            f.write(f"{mass} {x} {y} {vx} {vy}\n")
    
    return filename

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    
    n = int(sys.argv[1])
    generate(n)