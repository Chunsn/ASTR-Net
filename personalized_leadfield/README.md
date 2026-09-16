run\_pilot\_personalized\_leadfields.m



Purpose

This MATLAB script uses Brainstorm, SPM12, and OpenMEEG to generate subject-specific EEG leadfields standardized to 75 EEG channels and 994 cortical regions.



Requirements

\- MATLAB

\- Brainstorm with the SPM12 and OpenMEEG plugins

\- Brainstorm database: D:\\brainstorm\_db\\ASTR-NET

\- Environment variables: ASTR\_STANDARD\_MAT and ASTR\_INDEX\_FILE



Usage

Start Brainstorm in no-GUI mode, and then run the required processing stages:



brainstorm nogui

run\_pilot\_personalized\_leadfields('prepare')

run\_pilot\_personalized\_leadfields('mni\_usable')

run\_pilot\_personalized\_leadfields('headmodel\_usable')

run\_pilot\_personalized\_leadfields('reduce\_usable')



For subjects with repaired cortical surfaces, run:



run\_pilot\_personalized\_leadfields('mni\_recovered')

run\_pilot\_personalized\_leadfields('bem\_recovered')

run\_pilot\_personalized\_leadfields('headmodel\_recovered')

run\_pilot\_personalized\_leadfields('reduce\_recovered')



Output

outputs\\personalized\_leadfields\\<subject>\\fwd\_75x994.mat



Each output file contains the 75-by-994 leadfield matrix (fwd), its common-average-referenced version (fwd\_car), the sensor transformation matrix, and quality-control metadata.



