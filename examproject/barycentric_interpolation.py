import numpy as np

class barycentric_interpolation():

    def __init__(self):
        pass

    def eval(self,x1,x2,best,best_coordinates,y):
        # Function that takes two new coordinates, the old best (lowest) objective function value and the old coordinates that constituted it
        # If the two new coordinates constitute a lower objective function value, then the function should return these and the new (best) lowest objective function value
        # If not, the old (and yet best) two coordinates and their corresponding objective function value should be returned

        # Compute objective function for two new coordinates
        obj = np.sqrt((x1-y[0])**2+(x2-y[1])**2)

        # Evaluate whether new objective function is lower than the best so far
        if obj < best:

            # Update best objective function and coordinates with the new
            best_new = obj
            best_xpair = (x1,x2)
    
        # If it is not the lowest so far, let the new best objective function be the old best and let the coordinates remain the same
        else:
            best_new = best
            best_xpair = best_coordinates

        # Return (yet) best x coordinates and objective function value
        return best_xpair, best_new
    
    def find_ABCD(self,X,y,algo_no_print=False):

        if algo_no_print != True:
            print(f'{"-"*120}\nFinding coordinates for A,B,C and D')

        obj_best = np.ones(4)*np.inf
        best_coordinates = np.zeros((4,2))

        # Loop over x1 and x2 values pairwise
        for x1,x2 in X:

            # Evalute whether the given conditions are fulfilled
            # The first element in the list is for A, second for B, third for C and fourth for D
            bools = [x1 > y[0] and x2 > y[1], x1 > y[0] and x2 < y[1], x1 < y[0] and x2 < y[1], x1 < y[0] and x2 > y[1]]

            # Loop over the list of booleans
            for i,b in enumerate(bools):

                # If the conditions are fulfilled, then proceed to evaluate and replace objective function value and coordinates
                if b:
                    best_coordinates[i,:], obj_best[i] = self.eval(x1,x2,obj_best[i],best_coordinates[i,:],y)
        
                # If not, it remains the same as the best coordinates and lowest (best) objective function value is stored in the global scope

        interpolation_fail = (best_coordinates == 0).any()
        if interpolation_fail:
            best_coordinates = np.ones((4,2))*np.NaN

        # Convert array of lists with best coordinates to a list of tuples with the coordinate pairs
        ABCD_coordinates = [tuple(i) for i in best_coordinates]

        if algo_no_print != True and interpolation_fail != True:
            print(f'A,B,C and D coordinates have been succesfully found:')
            for i,name in enumerate(['A','B','C','D']):
                    print(f'{name}: ({best_coordinates[i][0]:.2f},{best_coordinates[i][1]:.2f})')
        
        if algo_no_print != True and interpolation_fail == True:
            print(f'Interpolation has failed')
            
        print(f'{"-"*120}')

        return ABCD_coordinates

    def compute_barycentric_coordinates(self,A,B,C,y,algo_no_print=False):

        # Compute the denominator since it is the same for both fractions for r1ABC and r2ABC
        denom = (B[1]-C[1])*(A[0]-C[0])+(C[0]-B[0])*(A[1]-C[1])
        r1ABC = ((B[1]-C[1])*(y[0]-C[0])+(C[0]-B[0])*(y[1]-C[1]))/(denom)
        r2ABC = ((C[1]-A[1])*(y[0]-C[0])+(A[0]-C[0])*(y[1]-C[1]))/(denom)
        r3ABC = 1 - r1ABC - r2ABC

        if algo_no_print != True:
            print(f'{"-"*120}\nFinding barycentric coordinates for the triangle')

        # Store barycentric coordinates in a list of coordinates
        list_of_r = [r1ABC,r2ABC,r3ABC]

        # Loop through barycentric coordinates and append the boolean for whether the barycentric coordinat is between 0 and 1
        list_of_r_bool = [0 < i < 1 for i in list_of_r]

        # Access whether all the booleans are true
        is_in_triangle = all(list_of_r_bool)

        if algo_no_print != True:
            print(f'Barycentric coordinates have been estimated:')
            print(f'r1ABC = {r1ABC:.2f}')
            print(f'r2ABC = {r2ABC:.2f}')
            print(f'r3ABC = {r3ABC:.2f}')

            if is_in_triangle:
                print(f'y is in the triangle')

            else:
                print(f'y is NOT in the triangle')
            
            print(f'{"-"*120}')

        return list_of_r, is_in_triangle

    def barycentric_algorithm(self,X,y,function,detailed_print=False):

        algo_no_print = True

        # 1. Use find_ABCD function to find coordinates and store them in coordinates_ABCD
        A,B,C,D = self.find_ABCD(X, y, algo_no_print)

        #print(f'{"-"*120}')

        if detailed_print:
            print(f'Initializing barycentric interpolation for coordinates y = ({y[0]:.2f},{y[1]:.2f}) \nDetermining A,B,C and D coordinates by calling find_ABCD')

        if np.isnan(A[0]):
            print(f'A, B, C or D cant be determined, why the interpolation has failed')
            print(f'The most probable cause is that the y set isnt within any possible triangle of the set X')
            return np.NaN
        
        if detailed_print:
            print(f'A,B,C and D coordinates have been succesfully found \nComputing barycentric coordinates by calling compute_barycentric_coordinates')

        ABC_barycentric = self.compute_barycentric_coordinates(A,B,C,y,algo_no_print)
        CDA_barycentric = self.compute_barycentric_coordinates(C,D,A,y,algo_no_print)

        # 2. Check if y is in ABC
        if ABC_barycentric[1]:

            if detailed_print:
                print(f'y is found in the triangle ABC')
            
            # Unpack barycentric coordinates
            list_of_r = ABC_barycentric[0]

            # Compute interpolated function value for y
            result_interpolation = list_of_r[0]*function(A)+list_of_r[1]*function(B)+list_of_r[2]*function(C)

        # 3. Check if y is in CDA
        elif CDA_barycentric[1]:
            
            if detailed_print:
                print(f'y is found in the triangle CDA')

            # Unpack barycentric coordinates
            list_of_r = CDA_barycentric[0]

            # Compute interpolated function value for y
            result_interpolation = list_of_r[2]*function(C)+list_of_r[3]*function(D)+list_of_r[0]*function(A)

        print(f'Barycentric interpolation result for y = ({y[0]:.2f},{y[1]:.2f}) ---> {result_interpolation:.5f}     compared to f(y)=f({y[0]:.2f},{y[1]:.2f}) = {function(y):.5f}')
        print(f'Barycentric interpolation approximates f(y)=f({y[0]:.2f},{y[1]:.2f}) with {100-np.abs(result_interpolation/function(y)):.2f}% accuracy')

        print(f'{"-"*120}')

        return result_interpolation

