"""
Module for controlling 2dfdr.

The actual 2dfdr call is done from run_2dfdr(). Other functions provide
more convenient access for different tasks: reducing a file
(run_2dfdr_single), combining files (run_2dfdr_combine) or loading the GUI
(load_gui).

The visit_dir context manager temporarily changes the working directory to
the one in which the file to be reduced is.

The temp_imp_scratch context manager temporarily sets the IMP_SCRATCH
environment variable to a temporary directory, allowing 2dfdr to be run in
parallel without conflicts. This is becoming unnecessary with recent
versions of 2dfdr which are ok with multiple instances, but there is no
particular reason to take it out.

Could easily be spun off as an independent module for general use. The
only thing holding this back is the assumed use of FITSFile objects as
defined in sami.manager, in run_2dfdr_single.

Note this module currently exists in two forms in two branches of this
repository: the default branch is compatible with 2dfdr v5, and the aaorun
branch is compatible with 2dfdr v6. The original plan had been to phase
out support for 2dfdr v5 and merge aaorun into default, but it may be
better to merge them in a way that retains the v5 functionality.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import subprocess
import os
import tempfile
import re
from contextlib import contextmanager
import six
import shutil, shlex
import warnings
import socket
import glob
import datetime
import numpy as np
import multiprocessing

# Set up logging
from .. import slogging
log = slogging.getLogger(__name__)
log.setLevel(slogging.WARNING)
# log.enable_console_logging()

LOCKDIR = '2dfdrLockDir'

COMMAND_GUI = 'drcontrol'
COMMAND_REDUCE = 'aaorun'

# Check that 2dfdr is available.
try:
    with open(os.devnull, 'w') as dump:
        subprocess.call([COMMAND_REDUCE, 'help'], stdout=dump)
except (OSError, FileNotFoundError):
    error_message = (
        'Cannot find the 2dfdr executable ``{}``\n'.format(COMMAND_REDUCE)
        + 'Please ensure that 2dfdr is correctly installed.')
    #raise ImportError(error_message)
    warnings.warn(error_message,stacklevel=2)
    
try:
    assert len(os.environ["DISPLAY"]) > 0
except (AssertionError, TypeError):
    raise ImportError("2dfdr requires a working DISPLAY. If you are running remotely, try enabling X-forwarding.")

if six.PY2:
    # Python 2 doesn't have TemporaryDirectory, so we backport it here:
    @contextmanager
    def TemporaryDirectory():
        dir_name = tempfile.mkdtemp()
        yield dir_name
        shutil.rmtree(dir_name)
else:
    TemporaryDirectory = tempfile.TemporaryDirectory


def subprocess_call(command_line, t_max=1200., **kwargs):
    """Generic function to run a command in an asynchronous way, capturing STDOUT and returning it."""
    formatted_command = " ".join(command_line)
    log.info("async call: {}".format(formatted_command))
    log.debug("Starting async processs: %s", formatted_command)

    if log.isEnabledFor(slogging.DEBUG):
        # Get current working directory and it's contents:
        log.debug("CWD: %s", subprocess.check_output("pwd", shell=False, stderr=None, **kwargs))
        log.debug(subprocess.check_output("ls", shell=True, stderr=None, **kwargs))

    # Create a temporary file to store the output 
    f = tempfile.NamedTemporaryFile()
    # Run the command 
    p = subprocess.Popen(command_line, shell=False, stdout=f, stderr=None, **kwargs)
    try: 
        #print("Waiting for timeout...")
        # Cause the command to time out after t_max seconds 
        _, _ = p.communicate(timeout=t_max)
    except subprocess.TimeoutExpired as e:
        print("Timeout has occurred!")
        p.terminate()

    # Open the file we just made to retrieve the output so we can put it in stdout 
    lines = open(f.name).readlines()
    stdout = "".join(lines)
    f.close()

    log.debug("Output from command '%s'", formatted_command)
    if log.isEnabledFor(slogging.DEBUG):
        for line in stdout.splitlines():
            log.debug("   " + line)
    return stdout


def call_2dfdr_reduce(dirname, root=None, options=None, dummy=False):
    """Call 2dfdr in pipeline reduction mode using `aaorun`"""
    # Make a temporary directory with a unique name for use as IMP_SCRATCH
    with TemporaryDirectory() as imp_scratch:
        command_line = [COMMAND_REDUCE]
        if options is not None:
            command_line.extend(options)

        if dummy:
            print('#####################')
            print('2dfdr call options:')
            print(' '.join(command_line))
            print('#####################')
            print()
        else:
            # Set up the environment:
            environment = dict(os.environ)
            environment["IMP_SCRATCH"] = imp_scratch

            if log.isEnabledFor(slogging.DEBUG):
                with open("2dfdr_commands.txt", "a") as cmd_file:
                    cmd_file.write("\n[2dfdr_command]\n")
                    cmd_file.write("working_dir = {}\n".format(dirname))
                    cmd_file.write("command = {}\n".format(
                        " ".join(map(shlex.quote, command_line))))


            with directory_lock(dirname):
                # add some debug printing:
               # print('2dfdr call options:')
                command_aaorun = ' '.join(command_line)
                print(' '.join(command_line))
                tdfdr_stdout = subprocess_call(command_line, cwd=dirname, env=environment)
               # print(tdfdr_stdout)

            # @TODO: Make this work with various versions of 2dfdr.
            # Confirm that the above command ran to completion, otherwise raise an exception
            try:
                confirm_line = tdfdr_stdout.splitlines()[-2]
                assert (
                    re.search(r"Action \"EXIT\", Task \S+, completed.*", tdfdr_stdout) is not None or  # 2dfdr v6.28
                    re.match(r"Data Reduction command \S+ completed.", confirm_line) is not None       # 2dfdr v6.14
                )
            except (IndexError, AssertionError): #write 2dfdr failures to tdfdr_failure.txt
                #log.debug(tdfdr_stdout)
                 #on {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())+'\n\n')

                message = '{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())+" 2dfdr did not run to completion for command: %s" % " ".join(command_line)
                print("Error has occured! Should check "+root+"/tdfdr_failure.txt")
                print("   "+message)
                lock = multiprocessing.Lock()
                if os.path.exists(root+'/tdfdr_failure.txt'):
                    lines = []; messages =[]
                    with open(root+'/tdfdr_failure.txt', 'r') as fail:
                        lines = np.array([line.rstrip() for line in fail])
                        if any(message[20:].rstrip() in line[20:].rstrip() for line in lines):
                            for line in lines:
                                if message[20:].rstrip() == line[20:].rstrip():
                                    lines[np.where(line == lines)] = message
                            messages = lines
                        else:
                            messages = np.append(np.array(lines), np.array(message))
                else:
                    messages = [message]
                with lock:
                    with open(root+'/tdfdr_failure.txt', 'w') as fn:
                        for message in messages:
                            fn.write(message + '\n')
                #raise TdfdrException(message)

def call_2dfdr_gui(dirname, options=None):
    """Call 2dfdr in GUI mode using `drcontrol`"""
    # Make a temporary directory with a unique name for use as IMP_SCRATCH
    with TemporaryDirectory() as imp_scratch:
        command_line = [COMMAND_GUI]
        if options is not None:
            command_line.extend(options)

        # Set up the environment:
        environment = dict(os.environ)
        environment["IMP_SCRATCH"] = imp_scratch

        with directory_lock(dirname):
            subprocess.run(command_line, cwd=dirname, check=True, env=environment)


def load_gui(dirname, idx_file=None):
    """Load the 2dfdr GUI in the specified directory."""
    # TODO: combine call_2dfdr_gui and load_gui
    if idx_file is not None:
        options = [idx_file]
    else:
        options = None
    with TemporaryDirectory() as imp_scratch:
        command_line = [COMMAND_GUI]
        if options is not None:
            command_line.extend(options)

        # Set up the environment:
        environment = dict(os.environ)
        environment["IMP_SCRATCH"] = imp_scratch

        with directory_lock(dirname):
            subprocess.run(command_line, cwd=dirname, check=True, env=environment)


#    call_2dfdr_gui(dirname, options)
    return


def run_2dfdr_single(fits, idx_file, root=None, options=None, dummy=False):
    """Run 2dfdr on a single FITS file."""
    print('Reducing file:', fits.filename)

    import time #sree will remove this after testing
    start_time = time.time() #sree will remove this after testing

    if fits.ndf_class == 'BIAS':
        task = 'reduce_bias'
    elif fits.ndf_class == 'DARK':
        task = 'reduce_dark'
    elif fits.ndf_class == 'LFLAT':
        task = 'reduce_lflat'
    elif fits.ndf_class == 'MFFFF':
        task = 'reduce_fflat'
    elif fits.ndf_class == 'MFARC':
        task = 'reduce_arc'
    elif fits.ndf_class == 'MFSKY':
        task = 'reduce_sky'
    elif fits.ndf_class == 'MFOBJECT':
        task = 'reduce_object'
    else:
        raise ValueError('Unrecognised NDF_CLASS')
    out_dirname = fits.filename[:fits.filename.rindex('.')] + '_outdir'
    out_dirname_full = os.path.join(fits.reduced_dir, out_dirname)
    out_dirname_tmp = os.path.join('/tmp/', out_dirname) 
    if socket.gethostname()[0:3] != 'aat':
        out_dirname_tmp = out_dirname_full

    # Originally we directly save output files to out_dirname_full which however makes reduce_arc() task 60 times slower when using the machine at AAT (e.g. aatlxe). 
    # The reason is that hector home is a network mounted drive and frequent comunications by 2dfdr makes is very slow. 
    # Also note that at AAT machine 'export IMP_SCRATCH=/tmp' speeds up 2dfdr tasks in the same manner. Check .bashrc at aat

    if not os.path.exists(out_dirname_tmp):  
        os.makedirs(out_dirname_tmp)

    options_all = [task, fits.filename, '-idxfile', idx_file,
                   '-OUT_DIRNAME', out_dirname_tmp]

    if options is not None:
        options_all.extend(options)
    call_2dfdr_reduce(fits.reduced_dir,root=root, options=options_all, dummy=dummy)

    if socket.gethostname()[0:3] == 'aat':
        files = glob.glob(out_dirname_tmp + '/*')
        if not os.path.exists(out_dirname_full):
            os.makedirs(out_dirname_full)
        for file in files:
            file_name = os.path.basename(file)
            shutil.move(file, os.path.join(out_dirname_full,file_name))
        shutil.rmtree(out_dirname_tmp)

    print("-- running time for "+fits.filename+" %s seconds --- will remove this from tdfdr.py" % (time.time() - start_time))  #sree will remove this after testing
    print('')

    return '2dfdr Reduced file:' + fits.filename


def run_2dfdr_combine(input_path_list, output_path, idx_file, dummy=False):
    """Run 2dfdr to combine the specified FITS files."""
    if len(input_path_list) < 2:
        raise ValueError('Need at least 2 files to combine!')
    output_dir, output_filename = os.path.split(output_path)
    options = ['combine_image',
               ' '.join([os.path.relpath(input_path, output_dir)
                         for input_path in input_path_list]),
               '-COMBINEDFILE',
               output_filename,
               '-idxfile',
               idx_file]
    call_2dfdr_reduce(output_dir, options=options, dummy=dummy)


def cleanup():
    """Clean up 2dfdr crud."""
    log.warning("It is generally not safe to cleanup 2dfdr in any other way than interactively!")
    subprocess.call(['cleanup'], stdout=subprocess.DEVNULL)


@contextmanager
def directory_lock(working_directory):
    """Create a context where 2dfdr can be run that is isolated from any other instance of 2dfdr."""

    lockdir = os.path.join(working_directory, LOCKDIR)

    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    # Attempt to obtain a directory lock in this directory
    try:
        os.mkdir(lockdir)
    except OSError:
        # Lock directory already exists, i.e. another task is working here
        # Run away!
        raise LockException(
            'Directory locked by another process: ' + working_directory)
    else:
        assert os.path.exists(lockdir)
        log.debug("Lock Directory '{}' created".format(lockdir))
    try:
        yield
    finally:
        log.debug("Will delete lock directory '{}'".format(lockdir))
        os.rmdir(lockdir)


class TdfdrException(Exception):
    """Base 2dfdr exception."""
    pass


class LockException(Exception):
    """Exception raised when attempting to work in a locked directory."""
    pass
