with open('bus.txt') as fp:
    for line in fp:
        line = line.rstrip('\n')
        print line
        execfile('demo.py' --ifile line)
