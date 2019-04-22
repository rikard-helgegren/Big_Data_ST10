import csv

'''
    Reads CSV file and returns all data in a matrix.
    
    @Param:  filname including .csv
    @Return: List of all data
'''
def read_CSV(filename):
    data = []
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
        for row in spamreader:
            data.append(row[0].split(','))

    return data