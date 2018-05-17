from numpy import *

def compute_error(b,m,points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        y = points[i, 1]
        x = points[i, 0]
        b_gradient += -(2/n) * (y - (m_current * x + b_current))
        m_gradient += -(2/n) * x * (y - (m_current * x + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, initial_b, initial_m, learning_rate, iterations):
    b = initial_b
    m = initial_m
    for i in range(iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def main():
    #Storing data from data.txt into points *numpy method*
    points = genfromtxt("data.txt", delimiter=',')
    #Hyperparamters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    iterations = 1000
    [b, m] = gradient_descent_runner(array(points), initial_b, initial_m, learning_rate, iterations)
    print("b:",  b, "m: ", m, "error:", compute_error(b, m, points))

if __name__ == '__main__':
    main()
