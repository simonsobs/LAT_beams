## 5.2.0 (2026-08-28)

### Feat

- add error plots
- run with new unroll option and better snr est
- add option to return smoothed map
- add option to combin non overlapping values of a field

### Fix

- better plotting, proper normalization, sigma2
- np2 trapz compat
- LF fwhm swap
- dont just log 0 without mpi

## 5.1.0 (2026-08-17)

### Feat

- integrate soma beam modes

### Fix

- more robust gauss fit
- dont blow up at 0

## 5.0.0 (2026-08-14)


- reorganize fitting module

## 4.0.0 (2026-08-14)

### Feat

- switch to wing projection type fit for bessel functions and add covariance terms
- errorbar plotting
- better MPI handling
- stacking refactor
- add basic LF support
- add option to not apply fscale_fac
- add single det mode
- add numeric metasplit
- first pass of bessel cov
- add cross template script
- job opening
- add option to open jobs without loadable amans
- add error codes
- add left right split
- handle splits in jobdb

### Fix

- modernize splits amd loading
- more robust cuts
- use new sotodlib logger
- new relcal naming
- update logger colors to function
- disable oversampling
- allow metasplits and fix ordering
- handle lf ws
- less stringet ptp cut
- ws meta
- cft -> cfg typo
- add missing config options
- split left right into two splits
- backwards compat
- open missing jobs
- dont error on old jobs without split
- update comment
- skip splits not in config
- splits bugs
- splits paths
- append padded dets
- get splits from config

## 3.3.1 (2026-06-23)

### Fix

- dont allow more SVD modes than detectors

## 3.3.0 (2026-06-23)

### Feat

- use nominal fwhm as smoothing kernel when estimating center in make_map

## 3.2.1 (2026-06-12)

### Fix

- remove stray paste

## 3.2.0 (2026-06-11)

### Feat

- two stage fitting, better hits calc, rescale pars
- set powell based on band
- add more fine grained control over merging fields
- add option to get points other than fwhm
- modify prefroc config with out context
- bessel fitting updates. Use L-BFGS-B when SNR is lower and mask out radii where the profile goes negetive when wing fitting
- better thresholding and prevent smoothing from making points negetive
- add merge feature to auto_relplot
- much better plotting and some error handing
- allow for radial sum and small fix in fits loading
- add error
- add blazar
- streamline of stacking code
- add metasplit option
- fit split maps, this is currently implemented in a somewhat temporary way since make_source_map doesn't add splits to the jobdb
- deconvolve pixel window func

### Fix

- allow for all tubes but o6
- apply gauss offset when computing fwhm
- bbox_inches typo
- dont plot all dets
- compat with older python
- save properly and remove bad cuts
- better starting amp
- autoscale x
- swap row and col again
- remove debugging code
- remove unused imports
- dont error out when computing fwhm for fake map and small metasplit typing fix
- lots of small fixes from refactor
- only check centering on filter bin map and fix indexing
- allow logger to work without MPI
- null out extra at end
- don't overwrite fits when fitting splits
- set a default comm
- some logger bugs and add profiler for map fitter

### Perf

- slight lstsq improvements

## 3.1.1 (2026-05-29)

### Perf

- switch to LoggerAdapter

## 3.1.0 (2026-05-28)

### Feat

- switch to new leakage normalization and include tb
- add a sript to compare planet profiles
- add auto_relplot functions
- much improved beam summary script and remove outdated pointing summary
- first pass at T->E
- many performance and accuracy improvements to tod fitting
- new beam summary script
- don't buffer logger by default

### Fix

- dont exclude catagory if its passed in as None
- add LF bands
- actually pass comps to plotter
- dont hardcode gauss symmetry
- merge into db
- don't save proc aman if we arent saving

### Perf

- some tod fit streamlining
- speedup plotting of TODs and fits
- use numba for gaussian

## 3.0.0 (2026-05-04)

### Feat

- allow for modified jobdb path
- add ws? to try_all list
- add multipole skip and fix offset
- fit bessel wing in bins
- add window output and profile and window plots
- smoother wing and lower singular thresh on lstsq
- update stacking code for splits
- mark jobs as open when rerunning obs
- add ability to produce maps from detector split lists
- much more streamlined bessel wing procedure
- include encoders and pwv
- add script to get stats on the cross

### Fix

- make cross summary work
- dont error out when noise model fails for one OT
- add missing config options
- switch to updated bessel fit
- plotting fixes
- lots of mpi fixes
- tmp dont open obs
- tmp use gauss SA
- recenter splits, fix tqdm, remove extra print
- handle more errors
- some small typing fixes
- dont use mpi logger when nproc=1
- fixes to ml mapper from jobdb integration and coordinate change, also add profiling mode

## 2.0.0 (2026-04-07)

### Feat

- jobdb for ml mapper
- make bessel wing optional
- refactor map fitting code and update to new bessel beam
- allow for max dets in tod plot
- make epochs outer loop and add more info to plot title
- add script for ML beam map
- move mapmaking funcs into library
- dont buffer log without MPI
- move to global config system and make some small configuration changes
- add option to not make ML map
- switch to Ted's mutex and make logger aware of the mutex
- stack ML maps and fix thumbnail radius
- update for new fit formate and qol improvements
- get maps from jobdb, bugfixes in scattering beam, improvements to dr4 beam fitting, added 1D jv^2 bessel beam
- better forced center
- wing for bessel beam and some bugfixes
- performance improvements
- add profiler mode when making maps, specify thread algo, dont modify aman in place when mapmakin
- do chisq cut only once both sin and cos terms are computed
- relcal cuts
- mask in gauss fit, handle center instability in bessel beam, use seperate buffer once map is cropped
- local preproc dir
- bessel beam fitting
- add ML mapping option
- add Qr and Ur
- add more planets for pointing fits
- cleaner multipole expansion
- context manager for log levels
- cleaner map plotting
- better multipole fitting and cleaner workflow
- move argparse to common func
- full refactor of stacking code
- move more tools into beam utils
- switch to logger
- streamline job setup and pep8
- refactor to reduce shared mapping code
- move aman loading into library
- dr4 profile fitting
- scale det secs properly and add job memory feature
- add more flexible profile splits mechanism
- add ability to fit a subset of soures and do a cleaner fwhm estimation
- print useful diagnostics
- cleaner gaussian fits and relative paths
- scale min det time with mask size and put relative paths in database
- cleaner output paths
- new profile and window function script
- streamlined gaussian fitting plus bugfixes
- jobdb optimization and folder append
- add hierachal MPI
- remove multistep fitting and take filter pars in from config
- switch to sotodlib downsample and remove unused file
- add profiler mode
- move plotting into its own file
- refactor pointing fits to per obs parallelism and implement preprocess and jobdb
- serialize config and metadata info
- more flexible guassian fit
- jobdb for fits
- streamline plotting for fits
- add jobdb
- recenter mask on source with initial map
- use preproc for beam mapping
- add source map stacker
- refactor of beam summary code
- refactor beam fitting pipeline
- add R2 output
- pad output for det match compatibility, fix fwhm cut, and add chisq cut
- switch fwhm cut to be relative to the nominal
- add mpi, actually center on source, minor plotting improvements
- add option to do QU
- do snr cuts
- allow mapping multiple sources in one config
- take in pointing mode and use updated pointing model
- app option to not use source mask
- switch to matthews solid angle estimator
- add pointing model database output
- much more physically motivated pointing model and add offset fitting script
- 7 parameter model
- pointing model
- better plotting, folder structure, and error handling
- split paths based on time
- rcw38
- include reduced chisq
- remove option for lm fitter
- add azel crossings to output, improved hits estimation, and include distance from prior
- split tod and fp plotting directories
- include hits
- improvements to flagging and blind searches
- context with tod pointing fits
- add a more generic pointing fit code
- add scripts for lat sso sims
- start adding some quick tools for first light

### Fix

- fix paths in preproc
- only check source list if we have a source tag
- use mpilock only when we have multiple procs
- pad block mask
- save ivar, dont offset maps, set size based on mask
- allow for no logger
- only plot some dets and dont double filter cutoff
- use source centered coords for ML map
- split joblist before scatter
- map config pars correctly
- resolve errors from refactor
- switch to lazy strings when logging
- use source list for pointing fits and add missing import
- cleanup and interp
- PEP8
- remove debugging code
- check for clipping
- dont forcibly remove offset
- pep8
- dont try to set x0 for now and remove unwanted prints
- set map colorbar based only on plotted region
- typo in loading config, add force centering as a config option, improved weighting when filling in center ring
- dont try to save preproc with multiple mpi procs running
- updated preproc fucntion
- better pointing error handling
- pointing bugfixes
- bugfixes latest refactor
- normalize weights properly
- better plotting of beams
- miscentering fixes
- stacking bugfixes
- normaization fixed
- rank 0 needs to open h5 file
- remove unused fake job class
- pointing fitter logging fixes
- handle nans in radial profile
- more orderly logging
- remove unused imports
- many small typo fixes
- normalzie logplots to temperature map but keep non log plots in real units
- don't overwrite model profile with data profile
- pep8
- several bugfixes from pointing refactor
- remove unused params and update config with defaults before serialization
- some cleanup
- pass correct units
- lots of bigfixes from refactor
- streamline plotting and clean up output
- consider turnarounds to be a seperate hit
- proprely fft trim single detectors and remove limit from testing
- pep8
- cleaner filter and better error handling
- more error handling
- better handling off losing all dets with mpi and cleaner blind search
- convert to pW and wait after sqlite error
- bettwe lower end cuts
- offsetaxis slice
- robust to db file swaps
- offsetaxis when ds
- samps offsets
- do final thresh on detector noise not average noise
- handle fft ringing
- outplut smoothed map to the correct directory
- fixes to query, better handling of no fit case, handle no flagged dets, some printing of which obs we are on
- adjust plotting path
