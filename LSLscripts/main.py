import os
from sys import executable
from subprocess import Popen  #CREATE_NEW_CONSOLE
from pprint import pprint

def main():
    try:
        while True:
            command = input('->: ')
            cmd = command.split()
            if len(cmd) == 0:
                # No command or just a space entered
                print('No command entered (LEN 0)')
                continue
            if cmd[0].lower() == 'exit':
                # Exit command
                break
            if cmd[0].lower() == 'ls':
                # Print python scripts in cur dir command
                pprint([f for f in os.listdir('.') if f[-2:] == 'py'])
                continue
            if cmd[0].lower() == 'run':
                # Run command
                if len(cmd) == 1:
                    # No Python script to run supplied
                    print('No argument supplied (LEN 1)')
                    continue
                
                fileName = cmd[1] + '.py'
                # Check if given file exists
                if not os.path.isfile(fileName):
                    # Script not found
                    print('Script ' + fileName + ' not found (case esensitive)')
                    continue
                Popen([executable, fileName], shell=True)
    except KeyboardInterrupt:
        pass

    print('Goodbye')
    return 

if __name__ == '__main__':
   main()