% Copy this template configuration file to your VOT workspace.
% Enter the full path to the ECO repository root folder.

CFWCR_repo_path = 'your CFWCR path'

tracker_label = 'CFWCR';
tracker_command = generate_matlab_command('CFWCR_VOT()', {[CFWCR_repo_path, 'vot2017_trax']});
tracker_interpreter = 'matlab';
tracker_trax = true;
