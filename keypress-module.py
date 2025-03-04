"""
KeyPressModule - Simple keyboard input handling for Tello drone control
"""

import pygame

def init():
    """Initialize the keyboard module"""
    pygame.init()
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Keyboard Input Window")
    print("KeyPress Module initialized")

def getKey(keyName):
    """
    Check if a specific key is pressed
    
    Args:
        keyName: pygame key name to check
        
    Returns:
        True if the key is pressed, False otherwise
    """
    ans = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    
    if keyInput[myKey]:
        ans = True
        
    pygame.display.update()
    return ans

def main():
    """Test function to verify the module works"""
    init()
    while True:
        if getKey("LEFT"):
            print("Left key pressed")
        if getKey("RIGHT"):
            print("Right key pressed")
        if getKey("UP"):
            print("Up key pressed")
        if getKey("DOWN"):
            print("Down key pressed")
        if getKey("q"):
            break

if __name__ == "__main__":
    main()
