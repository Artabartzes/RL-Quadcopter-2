from datetime import datetime
import csv

class Log:
    """Generic Logging utility."""

    SEVERITY_ERROR  = 'Error'
    SEVERITY_HIGH   = 'High'
    SEVERITY_MEDIUM = 'Medium'
    SEVERITY_LOW    = 'Low'
    SEVERITY_INFO   = 'Info'

    def __init__(self, filename = None):
        """Initialize parameters and opens log file."""
        if filename == None:
            filename = 'log_' + str(datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv'
        labels = ['DateTime', 'Id', 'Severity', 'Message']
        self.log = open(filename, 'w', newline='\n')
        self.logwriter = csv.writer(self.log)
        self.logwriter.writerow(labels)
        self.id_no = 0

    def __del__(self):
        """Class destructor closes the file reference."""
        self.close()

    def write(self, message, severity=None):
        """Writes a row to the log file."""
        if severity == None:
            severity = Log.SEVERITY_INFO
        dt = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.id_no +=1
        self.logwriter.writerow([dt] + [self.id_no] + [severity] + [message])

    def close(self):
        """Class destructor closes the file reference."""
        try:
            self.log.close()
        except IOError:
            print('An I/O error occured trying to close the file.')
        except:
            print('An unknown error occured trying to close the file.')
