function run_pilot_personalized_leadfields(stage)
% RUN_PILOT_PERSONALIZED_LEADFIELDS
% Reproducible pilot for sub02, sub1, sub29.
% Run after `brainstorm nogui` has started. Outputs are written only under

if nargin < 1
    stage = 'prepare';
end

cfg.root       = getenv('ASTR_WORKSPACE');
if isempty(cfg.root), cfg.root = pwd; end
cfg.db          = 'D:\brainstorm_db\ASTR-NET';
cfg.standardMat = getenv('ASTR_STANDARD_MAT');
cfg.indexFile   = getenv('ASTR_INDEX_FILE');
if isempty(cfg.standardMat) || isempty(cfg.indexFile)
    error('Set ASTR_STANDARD_MAT and ASTR_INDEX_FILE before running this pilot.');
end
cfg.output      = fullfile(cfg.root, 'outputs', 'personalized_leadfields');
cfg.subjects    = {'sub02', 'sub1', 'sub29'};
cfg.templateCortex = fullfile(cfg.db, 'anat', '@default_subject', 'tess_cortex_pial_low.mat');
cfg.templateMri    = fullfile(cfg.db, 'anat', '@default_subject', 'subjectimage_T1.mat');

if ~exist(cfg.output, 'dir'), mkdir(cfg.output); end

switch lower(stage)
    case 'prepare'
        prepare_template_and_sensors(cfg);
    case 'mni'
        prepare_template_and_sensors(cfg);
        map_all_subjects_to_994(cfg);
    case 'headmodel'
        compute_all_headmodels(cfg);
    case 'headmodel_good'
        cfg.subjects = {'sub02', 'sub1'};
        compute_all_headmodels(cfg);
    case 'headmodel_sub02'
        cfg.subjects = {'sub02'};
        compute_all_headmodels(cfg);
    case 'fix_sub02_channels'
        prepare_sub02_derived_channel(cfg);
    case 'fix_coordinate_repair_channels'
        repairSubjects = coordinate_repair_subjects();
        for i = 1:numel(repairSubjects)
            prepare_derived_channel_from_bids(cfg, repairSubjects{i});
        end
    case 'resegment_sub29'
        resegment_sub29_with_brainsuite();
    case 'reduce'
        reduce_all_headmodels(cfg);
    case 'reduce_good'
        cfg.subjects = {'sub02', 'sub1'};
        reduce_all_headmodels(cfg);
    case 'prepare_usable'
        cfg.subjects = usable_subjects();
        prepare_template_and_sensors(cfg);
    case 'mni_usable'
        cfg.subjects = usable_subjects();
        prepare_template_and_sensors(cfg);
        map_all_subjects_to_994(cfg);
    case 'mni_one'
        sub = getenv('ASTR_SUBJECT');
        if isempty(sub), error('Set ASTR_SUBJECT before running stage mni_one.'); end
        if ~ismember(sub, usable_subjects())
            error('%s is not in the structurally usable cohort.', sub);
        end
        cfg.subjects = {sub};
        prepare_template_and_sensors(cfg);
        map_all_subjects_to_994(cfg);
    case 'headmodel_usable'
        cfg.subjects = usable_subjects();
        compute_all_headmodels(cfg);
    case 'headmodel_missing_usable'
        cfg.subjects = usable_subjects();
        cfg.skipExistingHeadmodels = true;
        compute_all_headmodels(cfg);
    case 'reduce_usable'
        cfg.subjects = usable_subjects();
        reduce_all_headmodels(cfg);
    case 'mni_recovered'
        cfg.subjects = recovered_subjects();
        prepare_template_and_sensors(cfg);
        map_all_subjects_to_994(cfg);
    case 'headmodel_recovered'
        cfg.subjects = recovered_subjects();
        compute_all_headmodels(cfg);
    case 'bem_recovered'
        cfg.subjects = recovered_subjects();
        generate_recovered_bem(cfg);
    case 'reduce_recovered'
        cfg.subjects = recovered_subjects();
        reduce_all_headmodels(cfg);
    case 'all_recovered'
        cfg.subjects = recovered_subjects();
        prepare_template_and_sensors(cfg);
        map_all_subjects_to_994(cfg);
        compute_all_headmodels(cfg);
        reduce_all_headmodels(cfg);
    case 'all'
        prepare_template_and_sensors(cfg);
        map_all_subjects_to_994(cfg);
        compute_all_headmodels(cfg);
        reduce_all_headmodels(cfg);
    otherwise
        error(['Unknown stage: %s. Use prepare, mni, headmodel, headmodel_good, headmodel_sub02, ', ...
               'fix_sub02_channels, resegment_sub29, reduce, reduce_good, prepare_usable, ', ...
               'mni_usable, mni_one, headmodel_usable, headmodel_missing_usable, reduce_usable, ', ...
               'mni_recovered, bem_recovered, headmodel_recovered, reduce_recovered, ', ...
               'all_recovered, or all.'], stage);
end
end

function subjects = recovered_subjects()
% Subjects whose cortex was repaired and independently checked at 20,484 vertices.
subjects = {'sub05','sub17','sub21','sub29','sub31'};
end

function generate_recovered_bem(cfg)
% Rebuild the three BEM envelopes after corrected cortex import. The old BEM
% surfaces of the repaired subjects were generated from truncated cortexes.
for i = 1:numel(cfg.subjects)
    sub = cfg.subjects{i};
    ensure_subject_default_cortex(cfg, sub);
    fprintf('BEM> %s: regenerating 1922-vertex scalp/skull/brain envelopes.\n', sub);
    bst_report('Start');
    bst_process('CallProcess', 'process_generate_bem', [], [], ...
        'subjectname', sub, ...
        'nscalp',      1922, ...
        'nouter',      1922, ...
        'ninner',      1922, ...
        'thickness',   4, ...
        'method',      'brainstorm');
    reportFile = bst_report('Save');
    outDir = fullfile(cfg.output, sub);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    if ~isempty(reportFile)
        copyfile(reportFile, fullfile(outDir, 'bem_process_report.mat'));
    end
end
end

function subjects = usable_subjects()
% Excluded after structural QC: missing or grossly truncated cortex/MRI.
allSubjects = [{'sub1'}, arrayfun(@(i)sprintf('sub%02d', i), 2:36, 'UniformOutput', false)];
excluded = {'sub05','sub17','sub21','sub29','sub31'};
subjects = setdiff(allSubjects, excluded, 'stable');
end

function resegment_sub29_with_brainsuite()
bst_set('BrainSuiteDir', 'D:\BrainSuite21a');
[sSubject, iSubject] = bst_get('Subject', 'sub29');
if isempty(sSubject), error('Brainstorm subject sub29 was not found.'); end
fprintf('SEGMENT> Re-running BrainSuite cortical extraction and SVReg for sub29.\n');
[isOk, errMsg] = process_segment_brainsuite('Compute', iSubject, [], 15000, 0);
if ~isOk
    error('sub29 BrainSuite segmentation failed: %s', errMsg);
end
fprintf('SEGMENT> sub29 BrainSuite segmentation completed. %s\n', errMsg);
end

function prepare_sub02_derived_channel(cfg)
prepare_derived_channel_from_bids(cfg, 'sub02');
end

function prepare_derived_channel_from_bids(cfg, sub)
outDir = fullfile(cfg.output, sub);
if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, 'channel_avg_run01_eeg_world2scs_scalpprojected.mat');
if exist(outFile, 'file')
    fprintf('CHANNEL> Reusing %s\n', outFile); return;
end
[~, mriFile] = anatomy_files(cfg, sub);
channelFile = canonical_channel_file(cfg, sub);
scalpFile = fullfile(cfg.db, 'anat', sub, 'tess_head_bem_1922V.mat');
id = regexprep(sub, '^sub0?', '');
rawFile = fullfile('E:\ccepcoreg-bids\ccepcoreg-bids\ccepcoreg\derivatives\epochs', ...
    sprintf('sub-%02d', str2double(id)), 'eeg', sprintf('sub-%02d_task-ccepcoreg_electrodes.tsv', str2double(id)));
if ~exist(rawFile, 'file'), error('Missing original BIDS electrodes: %s', rawFile); end
raw = readtable(rawFile, 'FileType', 'text', 'Delimiter', '\t');
Mri = in_mri_bst(mriFile);
T = load(scalpFile, 'Vertices');
C = load(channelFile, 'Channel');
rawNames = lower(string(raw.name));
before = zeros(numel(C.Channel),3);
afterWorld = zeros(numel(C.Channel),3);
projected = zeros(numel(C.Channel),3);
dist = zeros(numel(C.Channel),1);
for i = 1:numel(C.Channel)
    name = lower(string(C.Channel(i).Name));
    j = find(rawNames == name, 1);
    if isempty(j), error('Original BIDS electrode %s is missing.', name); end
    before(i,:) = C.Channel(i).Loc(:)';
    pWorld = [raw.x(j), raw.y(j), raw.z(j)] ./ 1000;
    afterWorld(i,:) = cs_convert(Mri, 'world', 'scs', pWorld);
    [idx, dist(i)] = knnsearch(T.Vertices, afterWorld(i,:));
    projected(i,:) = T.Vertices(idx,:);
    C.Channel(i).Loc = projected(i,:)';
end
Channel = C.Channel; %#ok<NASGU>
channelQC = struct('method','BIDS T1w world->SCS followed by nearest scalp BEM vertex', ...
    'medianProjectionMm',median(dist)*1000, 'p95ProjectionMm',prctile(dist,95)*1000, ...
    'maxProjectionMm',max(dist)*1000, 'sourceChannelFile',channelFile, ...
    'sourceElectrodesFile',rawFile, 'scalpFile',scalpFile);
save(outFile, 'Channel', 'before', 'afterWorld', 'projected', 'dist', 'channelQC', '-v7');
fprintf('CHANNEL> %s derived channels saved; projection median/p95/max = %.2f/%.2f/%.2f mm.\n', sub, ...
    channelQC.medianProjectionMm, channelQC.p95ProjectionMm, channelQC.maxProjectionMm);
end

function prepare_template_and_sensors(cfg)
% Transfer the old 994 atlas with registered spheres, never with the old
% physical Vertices field. Build W: selected patient 75 sensors -> eloc75.
for i = 1:numel(cfg.subjects)
    if ismember(cfg.subjects{i}, coordinate_repair_subjects())
        prepare_derived_channel_from_bids(cfg, cfg.subjects{i});
    end
end
outFile = fullfile(cfg.output, 'template_994_and_sensor_setup_v3.mat');
if exist(outFile, 'file')
    existing = load(outFile, 'sensorSetup');
    needsUpdatedSensors = any(cellfun(@(sub) ~isfield(existing.sensorSetup, sub) || ...
        ~strcmp(existing.sensorSetup.(sub).channelFile, sensor_channel_file(cfg, sub)), cfg.subjects));
    if isfield(existing, 'sensorSetup') && ~needsUpdatedSensors
        fprintf('SETUP> Reusing %s\n', outFile);
        return;
    end
    % The atlas transfer is invariant; only append missing subject-specific
    % sensor transforms so existing validated template fields remain intact.
    sensorSetup = existing.sensorSetup;
    selectedNumbers = parse_selected_numbers(cfg.indexFile);
    for i = 1:numel(cfg.subjects)
        sub = cfg.subjects{i};
        if isfield(sensorSetup, sub) && strcmp(sensorSetup.(sub).channelFile, sensor_channel_file(cfg, sub)), continue; end
        chFile = sensor_channel_file(cfg, sub);
        C = load(chFile, 'Channel');
        [P, patientLabels] = select_channels_by_name(C.Channel, selectedNumbers);
        W = spherical_local_weights(P, load_standard_xyz(cfg.standardMat), 4);
        sensorSetup.(sub) = struct('channelFile', chFile, 'selectedLabels', {patientLabels}, ...
            'selectedXYZ', P, 'W_selected_to_standard', W, 'conditionNumber', cond(W), ...
            'rowSumError', max(abs(sum(W,2) - 1)));
    end
    save(outFile, 'sensorSetup', '-append');
    fprintf('SETUP> Appended sensor transforms for %d requested subjects: %s\n', numel(cfg.subjects), outFile);
    return;
end

std = load(cfg.standardMat, 'eloc75', 'icbm152', 'fwd');
tpl = load(cfg.templateCortex, 'Vertices', 'Faces', 'VertConn', 'Reg');
old = std.icbm152;

oldLabels = scouts_to_labels(old.Atlas(2).Scouts, size(old.Vertices,1));
[oldComp, nOld] = components_from_conn(old.VertConn);
[tplComp, nTpl] = components_from_surface(tpl, size(tpl.Vertices,1));
if nOld ~= 2 || nTpl ~= 2
    error('Expected two hemispheric components; got old=%d template=%d.', nOld, nTpl);
end

% Test both hemisphere pairings. The correct pairing preserves all 994 labels.
pairings = [1 2; 2 1];
best = struct('nLabels', -1);
for p = 1:size(pairings,1)
    labels = -ones(size(tpl.Vertices,1),1);
    dAll = [];
    for h = 1:2
        iOld = find(oldComp == h);
        iTpl = find(tplComp == pairings(p,h));
        qOld = unit_rows(old.Reg.Sphere.Vertices(iOld,:));
        qTpl = unit_rows(tpl.Reg.Sphere.Vertices(iTpl,:));
        [idx, d] = knnsearch(qOld, qTpl);
        labels(iTpl) = oldLabels(iOld(idx));
        dAll = [dAll; d]; %#ok<AGROW>
    end
    nLabels = numel(unique(labels));
    if nLabels > best.nLabels
        best.labels = labels;
        best.pairing = pairings(p,:);
        best.nLabels = nLabels;
        best.medianSphereDistance = median(dAll);
        best.maxSphereDistance = max(dAll);
    end
end
if best.nLabels ~= 994 || any(best.labels < 0)
    error('Template atlas transfer failed: only %d/994 labels retained.', best.nLabels);
end

selectedNumbers = parse_selected_numbers(cfg.indexFile);
if numel(selectedNumbers) ~= 75 || numel(unique(selectedNumbers)) ~= 75
    error('Expected 75 unique selected electrodes, got %d.', numel(selectedNumbers));
end
standardLabels = cellstr(string({std.eloc75.labels}));
standardXYZ = [[std.eloc75.X]' [std.eloc75.Y]' [std.eloc75.Z]'] ./ 1000;

% Sensor transform is computed independently for each head because positions
% are patient-specific. A global spherical spline was numerically ill-conditioned
% for this sparse selected subset, so use stable local spherical interpolation.
% The standard montage is expressed in SCS convention: x=anterior, y=left,
% z=superior, matching Brainstorm CTF SCS.
sensorSetup = struct();
for i = 1:numel(cfg.subjects)
    sub = cfg.subjects{i};
    chFile = sensor_channel_file(cfg, sub);
    C = load(chFile, 'Channel');
    [P, patientLabels] = select_channels_by_name(C.Channel, selectedNumbers);
    W = spherical_local_weights(P, standardXYZ, 4);
    sensorSetup.(sub).channelFile = chFile;
    sensorSetup.(sub).selectedLabels = patientLabels;
    sensorSetup.(sub).selectedXYZ = P;
    sensorSetup.(sub).W_selected_to_standard = W;
    sensorSetup.(sub).conditionNumber = cond(W);
    sensorSetup.(sub).rowSumError = max(abs(sum(W,2) - 1));
end

templateLabels = best.labels; %#ok<NASGU>
templateHemisphere = orient_hemisphere_ids(tplComp, tpl.Vertices); %#ok<NASGU>
templateVertices = tpl.Vertices; %#ok<NASGU>
templateFaces = tpl.Faces; %#ok<NASGU>
standardFwd = std.fwd; %#ok<NASGU>
save(outFile, 'templateLabels', 'templateHemisphere', 'templateVertices', 'templateFaces', ...
    'standardLabels', 'standardXYZ', 'standardFwd', 'selectedNumbers', 'sensorSetup', 'best', '-v7.3');
fprintf('SETUP> Saved template atlas and three sensor transforms: %s\n', outFile);
end

function map_all_subjects_to_994(cfg)
setup = load(fullfile(cfg.output, 'template_994_and_sensor_setup_v3.mat'));
tplMri = in_mri_bst(cfg.templateMri);
templateMni = cs_convert(tplMri, 'scs', 'mni', setup.templateVertices);
if any(~isfinite(templateMni(:)))
    error('Template SCS->MNI conversion produced invalid coordinates.');
end

for i = 1:numel(cfg.subjects)
    sub = cfg.subjects{i};
    outDir = fullfile(cfg.output, sub);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    outFile = fullfile(outDir, 'atlas_994_native_v2.mat');
    if exist(outFile, 'file')
        fprintf('MNI> Reusing %s\n', outFile);
        continue;
    end
    [cortexFile, mriFile] = anatomy_files(cfg, sub);
    % Imported/reduced corrected surfaces may omit VertConn; Faces is the
    % deterministic fallback used to reconstruct mesh connectivity.
    T = load(cortexFile, 'Vertices', 'VertConn', 'Faces');
    [patientHemisphere, nComp] = components_from_surface(T, size(T.Vertices,1));
    patientHemisphere = orient_hemisphere_ids(patientHemisphere, T.Vertices);
    if nComp < 2
        error('%s cortex does not contain two hemispheric components.', sub);
    end
    sMri = in_mri_bst(mriFile);
    if ~has_nonlinear_mni(sMri)
        fprintf('MNI> %s: computing non-linear SPM normalization (not writing the original MRI file).\n', sub);
        [sMri, errMsg] = bst_normalize_mni(sMri, 'segment');
        if ~isempty(errMsg) || isempty(sMri) || ~has_nonlinear_mni(sMri)
            error('%s MNI normalization failed: %s', sub, errMsg);
        end
    else
        fprintf('MNI> %s: reusing existing non-linear deformation fields.\n', sub);
    end
    warpedTemplate = cs_convert(sMri, 'mni', 'scs', templateMni);
    valid = all(isfinite(warpedTemplate), 2);
    if nnz(valid) < 0.99 * size(warpedTemplate,1)
        error('%s: %.1f%% template vertices invalid after inverse MNI warp.', sub, 100*(1-nnz(valid)/numel(valid)));
    end
    [regionLabels, nnDistance] = transfer_labels_by_hemisphere( ...
        T.Vertices, patientHemisphere, warpedTemplate, setup.templateHemisphere, setup.templateLabels, valid);
    [regionLabels, rescueCount] = guarantee_all_regions( ...
        T.Vertices, patientHemisphere, warpedTemplate, setup.templateHemisphere, setup.templateLabels, regionLabels);
    regionCounts = accumarray(regionLabels + 1, 1, [994 1]);
    if any(regionCounts == 0)
        error('%s: %d atlas regions are empty after rescue.', sub, nnz(regionCounts==0));
    end
    mappingQC = struct('subject',sub, 'nVertices',size(T.Vertices,1), ...
        'medianNearestDistanceMm',median(nnDistance)*1000, ...
        'p95NearestDistanceMm',prctile(nnDistance,95)*1000, ...
        'maxNearestDistanceMm',max(nnDistance)*1000, ...
        'regionCountMin',min(regionCounts), 'regionCountMedian',median(regionCounts), ...
        'regionCountMax',max(regionCounts), 'rescuedEmptyRegions',rescueCount, ...
        'nonlinearMni',true, 'cortexFile',cortexFile, 'mriFile',mriFile);
    patient_region_labels = regionLabels; %#ok<NASGU>
    save(outFile, 'patient_region_labels', 'warpedTemplate', 'mappingQC', 'regionCounts', '-v7.3');
    fprintf('MNI> %s completed: median/p95 nearest distance %.2f/%.2f mm; rescue=%d.\n', ...
        sub, mappingQC.medianNearestDistanceMm, mappingQC.p95NearestDistanceMm, rescueCount);
end
end

function compute_all_headmodels(cfg)
% Generates the physical OpenMEEG gain in the Brainstorm database. Final reduced
% matrices remain in cfg.output. Do not proceed if the 994 atlas step failed.
for i = 1:numel(cfg.subjects)
    sub = cfg.subjects{i};
    ensure_subject_default_cortex(cfg, sub);
    if ~exist(fullfile(cfg.output, sub, 'atlas_994_native_v2.mat'), 'file')
        error('%s: atlas_994_native_v2.mat is missing. Run stage mni first.', sub);
    end
    dataFile = canonical_data_file(cfg, sub);
    if isfield(cfg, 'skipExistingHeadmodels') && cfg.skipExistingHeadmodels && ...
            ~isempty(dir(fullfile(cfg.db, 'data', sub, '**', 'headmodel_surf_openmeeg*.mat')))
        fprintf('HEADMODEL> %s: reusing existing OpenMEEG headmodel.\n', sub);
        continue;
    end
    restoreChannel = [];
    if ismember(sub, coordinate_repair_subjects())
        derivedFile = fullfile(cfg.output, sub, 'channel_avg_run01_eeg_world2scs_scalpprojected.mat');
        if ~exist(derivedFile, 'file'), error('Run fix_coordinate_repair_channels before computing %s headmodel.', sub); end
        channelFile = canonical_channel_file(cfg, sub);
        backupFile = fullfile(cfg.output, sub, 'channel_avg_run01_eeg_original_backup.mat');
        if ~exist(backupFile, 'file'), copyfile(channelFile, backupFile); end
        D = load(derivedFile, 'Channel');
        Channel = D.Channel; %#ok<NASGU>
        save(channelFile, 'Channel', '-append');
        restoreChannel = onCleanup(@()copyfile(backupFile, channelFile, 'f'));
        bst_memory('UnloadAll', 'Forced');
        fprintf('HEADMODEL> %s: using temporary BIDS-world-to-SCS scalp-projected channel coordinates.\n', sub);
    end
    fprintf('HEADMODEL> %s: computing/recomputing OpenMEEG BEM gain.\n', sub);
    bst_report('Start');
    bst_process('CallProcess', 'process_headmodel', dataFile, [], ...
        'sourcespace', 1, ...
        'eeg',         3, ...
        'openmeeg', struct( ...
            'BemSelect',    [0, 0, 1], ...
            'BemCond',      [1, 0.0125, 1], ...
            'BemNames',     {{'Scalp', 'Skull', 'Brain'}}, ...
            'BemFiles',     {{}}, ...
            'isAdjoint',    1, ...
            'isAdaptative', 1, ...
            'isSplit',      0, ...
            'SplitLength',  4000));
    reportFile = bst_report('Save');
    if ~isempty(reportFile)
        copyfile(reportFile, fullfile(cfg.output, sub, 'headmodel_process_report.mat'));
    end
    clear restoreChannel
end
end

function reduce_all_headmodels(cfg)
setup = load(fullfile(cfg.output, 'template_994_and_sensor_setup_v3.mat'), 'sensorSetup', 'standardFwd', 'standardLabels');
for i = 1:numel(cfg.subjects)
    sub = cfg.subjects{i};
    atlasFile = fullfile(cfg.output, sub, 'atlas_994_native_v2.mat');
    if ~exist(atlasFile, 'file'), error('%s atlas mapping missing.', sub); end
    [~, dataFile] = anatomy_files(cfg, sub);
    %#ok<NASGU> % anatomy_files validates the subject; channel file below is canonical
    chFile = sensor_channel_file(cfg, sub);
    hmFile = newest_headmodel_for_subject(cfg, sub);
    H = load(hmFile, 'Gain', 'GridLoc', 'GridOrient', 'Comment', 'EEGMethod');
    A = load(atlasFile, 'patient_region_labels', 'mappingQC');
    C = load(chFile, 'Channel');
    if size(H.Gain,1) ~= numel(C.Channel)
        error('%s: Gain has %d rows but channel file has %d channels.', sub, size(H.Gain,1), numel(C.Channel));
    end
    if size(H.GridLoc,1) ~= numel(A.patient_region_labels) || size(H.GridOrient,1) ~= numel(A.patient_region_labels)
        error('%s: headmodel/source atlas vertex mismatch.', sub);
    end
    [~, selectedLabels, selectedRows] = select_channels_by_name(C.Channel, parse_selected_numbers(cfg.indexFile));
    gainSelected = H.Gain(selectedRows,:);
    nVert = size(H.GridLoc,1);
    gainReduced = zeros(75, nVert);
    for v = 1:nVert
        cols = (v-1)*3 + (1:3);
        gainReduced(:,v) = gainSelected(:,cols) * H.GridOrient(v,:)';
    end
    regionalActual = zeros(75,994);
    for r = 0:993
        regionalActual(:,r+1) = mean(gainReduced(:, A.patient_region_labels == r), 2);
    end
    W = setup.sensorSetup.(sub).W_selected_to_standard;
    fwd = W * regionalActual;
    Hcar = eye(75) - ones(75)/75;
    fwd_car = Hcar * fwd;
    qc = struct('subject',sub, 'headmodelFile',hmFile, 'channelFile',chFile, ...
        'gainSize',size(H.Gain), 'fwdSize',size(fwd), 'allFinite',all(isfinite(fwd),'all'), ...
        'zeroColumns',nnz(vecnorm(fwd)==0), 'rawColumnMeanRatio',norm(mean(fwd,1))/norm(fwd,'fro'), ...
        'standardFwdStd',std(setup.standardFwd,0,'all'), 'patientFwdStd',std(fwd,0,'all'), ...
        'mappingQC',A.mappingQC);
    standard_channel_labels = setup.standardLabels; %#ok<NASGU>
    selected_patient_labels = selectedLabels; %#ok<NASGU>
    sensor_transform = W; %#ok<NASGU>
    save(fullfile(cfg.output, sub, 'fwd_75x994.mat'), 'fwd', 'fwd_car', 'regionalActual', ...
        'standard_channel_labels', 'selected_patient_labels', 'sensor_transform', 'qc', '-v7.3');
    fprintf('REDUCE> %s saved %s\n', sub, fullfile(cfg.output, sub, 'fwd_75x994.mat'));
end
end

function [labels, d] = transfer_labels_by_hemisphere(patientV, patientH, warpV, templateH, templateLabels, valid)
labels = -ones(size(patientV,1),1); d = zeros(size(patientV,1),1);
for h = 1:2
    ip = find(patientH == h);
    it = find(templateH == h & valid);
    [idx, d(ip)] = knnsearch(warpV(it,:), patientV(ip,:));
    labels(ip) = templateLabels(it(idx));
end
end

function [labels, nRescue] = guarantee_all_regions(patientV, patientH, warpV, templateH, templateLabels, labels)
nRescue = 0;
counts = accumarray(labels+1, 1, [994 1]);
for r = find(counts == 0)'
    r0 = r-1;
    ih = mode(templateH(templateLabels == r0));
    ip = find(patientH == ih);
    it = find(templateH == ih & templateLabels == r0 & all(isfinite(warpV),2));
    % Borrow only from a region with >1 vertices. This guarantees that
    % repairing a missing region never creates a new missing region.
    donors = ip(counts(labels(ip)+1) > 1);
    if isempty(donors)
        error('No donor vertex available while rescuing atlas region %d.', r0);
    end
    [nearest, d] = knnsearch(patientV(donors,:), warpV(it,:));
    [~, best] = min(d);
    chosen = donors(nearest(best));
    counts(labels(chosen)+1) = counts(labels(chosen)+1) - 1;
    labels(chosen) = r0;
    counts(r) = counts(r) + 1;
    nRescue = nRescue + 1;
end
end

function [comp, nComp] = components_from_conn(conn)
[comp, binSizes] = conncomp(graph(conn));
nComp = numel(binSizes);
comp = comp(:);
end

function [comp, nComp] = components_from_surface(T, nVert)
if isfield(T, 'VertConn') && ~isempty(T.VertConn)
    [comp, nComp] = components_from_conn(T.VertConn);
    return;
end
F = double(T.Faces);
ii = [F(:,1); F(:,2); F(:,3)];
jj = [F(:,2); F(:,3); F(:,1)];
conn = sparse([ii; jj], [jj; ii], 1, nVert, nVert);
[comp, nComp] = components_from_conn(conn);
end

function hemi = orient_hemisphere_ids(comp, vertices)
% Brainstorm SCS: positive y is left, negative y is right. conncomp IDs are
% arbitrary, therefore normalize them before any cross-subject mapping. A
% tiny disconnected mesh fragment can occur after cortical meshing; retain
% the two large hemispheres and assign only negligible fragments by nearest
% hemispheric mean-y. Larger extra components are a structural QC failure.
ids = unique(comp(:));
counts = arrayfun(@(x)nnz(comp==x), ids);
if numel(ids) < 2, error('Expected at least two components for hemisphere orientation.'); end
[sortedCounts, order] = sort(counts, 'descend');
major = ids(order(1:2));
minorCount = sum(sortedCounts(3:end));
if minorCount > 0.005 * numel(comp)
    error('Cortex has %d extra-component vertices (%.2f%%), exceeding the 0.5%% QC limit.', ...
        minorCount, 100*minorCount/numel(comp));
end
meanY = arrayfun(@(x)mean(vertices(comp==x,2)), major);
hemi = zeros(size(comp));
[~, leftIdx] = max(meanY);
leftId = major(leftIdx);
rightId = major(3-leftIdx);
hemi(comp == leftId) = 1;
hemi(comp == rightId) = 2;
for id = setdiff(ids, major)'
    y = mean(vertices(comp==id,2));
    if abs(y - meanY(leftIdx)) <= abs(y - meanY(3-leftIdx))
        hemi(comp==id) = 1;
    else
        hemi(comp==id) = 2;
    end
end
end

function labels = scouts_to_labels(scouts, nVert)
labels = -ones(nVert,1);
for r = 1:numel(scouts)
    ix = scouts(r).Vertices(:);
    labels(ix) = r-1;
end
if any(labels < 0), error('Standard 994 scouts do not cover all vertices.'); end
end

function [P, labels, rows] = select_channels_by_name(Channel, numbers)
names = cellstr(lower(string({Channel.Name})));
labels = arrayfun(@(x)sprintf('e%d',x), numbers, 'UniformOutput', false);
rows = zeros(numel(numbers),1);
P = zeros(numel(numbers),3);
for i = 1:numel(numbers)
    row = find(strcmp(names, labels{i}), 1);
    if isempty(row), error('Required channel %s was not found.', labels{i}); end
    rows(i) = row;
    P(i,:) = Channel(rows(i)).Loc(:)';
end
end

function W = spherical_local_weights(sourceXYZ, targetXYZ, nNeighbors)
source = unit_rows(sourceXYZ - mean(sourceXYZ,1));
target = unit_rows(targetXYZ - mean(targetXYZ,1));
W = zeros(size(target,1), size(source,1));
angles = acos(max(-1, min(1, target * source')));
for r = 1:size(target,1)
    [a, ix] = mink(angles(r,:), nNeighbors);
    % 2 degree floor prevents a nearly coincident contact dominating a row.
    w = 1 ./ (a.^2 + deg2rad(2)^2);
    W(r,ix) = w ./ sum(w);
end
end

function U = unit_rows(X)
U = X ./ vecnorm(X,2,2);
end

function xyz = load_standard_xyz(standardMat)
S = load(standardMat, 'eloc75');
xyz = [[S.eloc75.X]' [S.eloc75.Y]' [S.eloc75.Z]'] ./ 1000;
end

function tf = has_nonlinear_mni(sMri)
tf = isfield(sMri, 'NCS') && isfield(sMri.NCS, 'y') && isfield(sMri.NCS, 'iy') && ...
     ~isempty(sMri.NCS.y) && ~isempty(sMri.NCS.iy) && ...
     isfield(sMri.NCS, 'y_vox2ras') && ~isempty(sMri.NCS.y_vox2ras);
end

function nums = parse_selected_numbers(indexFile)
txt = fileread(indexFile);
tokens = regexp(lower(txt), 'e(\d+)', 'tokens');
nums = cellfun(@(c)str2double(c{1}), tokens);
end

function chFile = canonical_channel_file(cfg, sub)
chFile = fullfile(cfg.db, 'data', sub, 'avg_run01eeg', 'channel_avg_run01_eeg.mat');
if ~exist(chFile, 'file'), error('%s channel file missing: %s', sub, chFile); end
end

function chFile = sensor_channel_file(cfg, sub)
% Coordinate-repair subjects use a derived, physically corrected copy for both
% gain computation and the 75-channel interpolation.
derived = fullfile(cfg.output, sub, 'channel_avg_run01_eeg_world2scs_scalpprojected.mat');
if ismember(sub, coordinate_repair_subjects()) && exist(derived, 'file')
    chFile = derived;
else
    chFile = canonical_channel_file(cfg, sub);
end
end

function subjects = coordinate_repair_subjects()
subjects = {'sub02','sub25','sub34','sub36'};
end

function dataFile = canonical_data_file(cfg, sub)
dataFile = fullfile(cfg.db, 'data', sub, 'avg_run01eeg', 'data_avg_run01_eeg.mat');
if ~exist(dataFile, 'file'), error('%s data file missing: %s', sub, dataFile); end
end

function [cortexFile, mriFile] = anatomy_files(cfg, sub)
anatDir = fullfile(cfg.db, 'anat', sub);
corrected = struct('sub05','tess_cortex_pial_06.mat', ...
    'sub17','tess_cortex_pial_06.mat', ...
    'sub21','tess_cortex_pial_02.mat', ...
    'sub29','tess_cortex_pial_18.mat', ...
    'sub31','tess_cortex_pial_06.mat');
if isfield(corrected, sub)
    cortexFile = fullfile(anatDir, corrected.(sub));
else
    cortexFile = fullfile(anatDir, 'tess_cortex_pial_02.mat');
end
if ~exist(cortexFile, 'file'), error('%s cortex file missing: %s', sub, cortexFile); end
T = load(cortexFile, 'Vertices');
if size(T.Vertices,1) ~= 20484
    error('%s selected cortex has %d vertices instead of 20484: %s', ...
        sub, size(T.Vertices,1), cortexFile);
end
m = dir(fullfile(anatDir, 'subjectimage_*T1w.mat'));
if numel(m) ~= 1, error('%s expected one T1w MRI, found %d.', sub, numel(m)); end
mriFile = fullfile(m.folder, m.name);
end

function ensure_subject_default_cortex(cfg, sub)
% OpenMEEG uses the Brainstorm default cortex, so force it to the validated file.
[cortexFile, ~] = anatomy_files(cfg, sub);
[sSubject, iSubject] = bst_get('Subject', sub);
if isempty(sSubject), error('Brainstorm subject %s was not found.', sub); end
[~, base, ext] = fileparts(cortexFile);
relativeFile = strrep(fullfile(sub, [base ext]), '\', '/');
iSurface = find(strcmpi({sSubject.Surface.FileName}, relativeFile), 1);
if isempty(iSurface)
    error('%s validated cortex is not registered in Brainstorm: %s', sub, relativeFile);
end
if isempty(sSubject.iCortex) || sSubject.iCortex ~= iSurface
    db_surface_default(iSubject, 'Cortex', iSurface);
    fprintf('CORTEX> %s default set to %s\n', sub, relativeFile);
else
    fprintf('CORTEX> %s already uses %s\n', sub, relativeFile);
end
end

function hmFile = newest_headmodel_for_subject(cfg, sub)
d = dir(fullfile(cfg.db, 'data', sub, '**', 'headmodel_surf_openmeeg*.mat'));
if isempty(d), error('%s: no OpenMEEG surface headmodel found.', sub); end
[~, i] = max([d.datenum]);
hmFile = fullfile(d(i).folder, d(i).name);
end
