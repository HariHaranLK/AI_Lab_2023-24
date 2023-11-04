# Ex.No: 13 Learning - Use Supervised Learning
### DATE:28/10.2023
### REGISTER NUMBER: 212221040051
### AIM: 
To write the program to train the classifier for Simple delivery Robot Planner.
###  Algorithm:
Step 1: Determine the robot's current location in the warehouse. <br>
Step 2: Identify the goal location where the package needs to be delivered. <br>
Step 3: Create a map of the warehouse layout, marking obstacles and paths. <br>
Step 4: Use the A* search algorithm to find the optimal path from the robot's current location to the goal while avoiding obstacles. <br>
Step 5: Break the path into individual movement steps. <br>
Step 6: Check for the next step in the path and move the robot accordingly. <br>
Step 7: Continuously update the robot's location. <br>
Step 8: If the robot encounters an obstacle, reevaluate the path to find an alternative route. <br>
Step 9: Repeat steps 6-8 until the robot reaches the goal location. <br>
Step 10: Once the robot reaches the goal, complete the package delivery. <br>
### Program:
```
(define (domain delivery_domain)
  (:requirements :strips)
  (:predicates
    (at ?robot ?location)
    (has-package ?robot ?package)
    (at-goal ?package ?location)
    (clear ?location)
    (free ?robot)
    (package-at ?package ?location)
  )
  
  (:action move
    :parameters (?robot ?from ?to)
    :precondition (and (at ?robot ?from) (clear ?to) (free ?robot))
    :effect (and (at ?robot ?to) (clear ?from))
  )

  (:action pick-up
    :parameters (?robot ?package ?location)
    :precondition (and (at ?robot ?location) (package-at ?package ?location) (free ?robot))
    :effect (and (has-package ?robot ?package) (at ?robot ?location) (not (package-at ?package ?location)))
  )

  (:action drop-off
    :parameters (?robot ?package ?location)
    :precondition (and (at ?robot ?location) (has-package ?robot ?package) (at-goal ?package ?location) (free ?robot))
    :effect (and (package-at ?package ?location) (clear ?location) (free ?robot) (not (has-package ?robot ?package)))
  )
)

```
### Input 
```
(define (problem delivery_problem)
  (:domain delivery_domain)
  (:objects
    robot1
    package1
    locationA
    locationB
  )
  (:init
    (at robot1 locationA)
    (package-at package1 locationA)
    (at-goal package1 locationB)
    (clear locationB)
    (clear locationA)
    (free robot1)
  )
  (:goal (and (package-at package1 locationB) (clear locationB)))
)

```
### Output/Plan:

![image](https://github.com/HariHaranLK/AI_Lab_2023-24/assets/132996089/68eda73c-9e8d-4c9f-be6e-25d8be253dab)

### Result:
Thus the system was trained successfully and the prediction was carried out.
