tic

reloadIm = 1;

if reloadIm == 1
    clear
    close all
    
    reloadIm = 1;
end

%Updated 10/23/18
%V3, added histogram statistics
%Updated 1/25/19; added 4D plot

%Automated analyzer for real-time isothermal amplification & melt curve
%by Erik Jue, 2018

%This MATLAB script performs the following functions
%Loads a microscope .tif image sequence for digital LAMP experiment
%Loads a .txt file of the s came name containing time and temp data
%Uses first image frame (low temp) to detect total # wells
%Uses last frame of LAMP to detect positive wells
%Track the intensity of the positive wells over time
%Apply gaussian smoothing and baseline subtraction
%Saves the data
%Calculate intensity during melt curve
%Repeats for each tiff stack in the folder 

%Requires Bio-Formats 5.8.1 by OME
%https://www.openmicroscopy.org/bio-formats/
%Requires Control ZEN Blue and the microscope from MATLAB (ReadImage6D.m) by Sebastian Rhode
%https://www.mathworks.com/matlabcentral/fileexchange/50079-control-zen-blue-and-the-microscope-from-matlab?focused=6875595&tab=function
%Requires Create Video of Rotating 3D Plot 1.0.0.0 by Alan Jennings
%https://www.mathworks.com/matlabcentral/fileexchange/41093-create-video-of-rotating-3d-plot


%Notes specific to this data set
%Used a connectivity of 4 (cardinal directions only) for well detection
%Ignore last frame since it does not capture the full exposure
%Sample fluorescence is high at low temp, use this to detect all wells
%Sample fluorescence decreases with high temp, ignore all heating frames 

%% VARIABLES FOR EDITING

%FOLDER TO ANALYZE, WILL ANALYZE ALL .TIF FILES IT FINDS
%TIF FILES MUST HAV E MATCHING NAME TO TEMPDATA
folder = 'C:\Users\Lab User\Desktop\Digital_NAAT_2Ch_MeltCurve_Analyzer\exampleSet\'; %'C:\user\data';

%SAVE LOCATION
saveHere = 'C:\Users\Lab User\Desktop\Digital_NAAT_2Ch_MeltCurve_Analyzer\exampleSet\'; %'C:\user\results\';

%Saves workspace variables to a .mat file
saveVars = 1; 

%Adds the trailing \ if applicable
if saveHere(end)~='\'
    saveHere = strcat(saveHere, '\');
end

%SETTINGS, 1 means on, 0 means off
%Set to 1 to save images
imageSave = 1;
%Set to 1 to save summary info to excel file
excelSaveSummary = 1;
%Set to 1 to save intensity curves to excel file
excelSaveIntensity = 1;

%baselineAdjust averages the intensities between all frames in between 
%baselineStart and baselineEnd and subtracts this value for all frames
%Set to 1 to enable background subtraction
baselineAdjust = 1;

%This is the frame number when the sample reaches the correct temp
frameAtTemp = 8;

%slopeAdjust finds the slope between the baselineEnd frame and
%baselineStart frame and propagates the slope correction for all frames
%This can be used to account for slow drift of negatives wells
%Note: do not use if raw data is very noisy
%Set to 1 to enable background slope correction
slopeAdjust = 0;

%Well Detection Settings
%Multi-level thresholding
%This techniques applies a threshold to detect a set of wells and selects
%selects wells that fall in the areaBound and majorAxisBound. Multiple
%rounds of different thresholds are applied to detect a greater number of
%wells

%Manually determine threshold levels for the masking images
%Reducing the middle number will improve well detection resolution but 
%increase processing time. 
mask_thresh = .08:.002:.32; %Could be .002 (time scales linearly, .008=5min)

%lower and upper cutoff for the area of each well
%Units in # of pixels
areaBound = [20 45];

%Thresholding is used for automated time-to-positive determination
%Will calculate ttp after baselineAdjust and slopeAdjust if applicable
threshold = 250;

%A gaussian filter is used to smooth the intensity curves
%Set the gaussian window smoothing size
gaussWinSize = 5;

%Maximum allowable slope to detect
maxSlope = 200;

%Threshold for the max slope required to call a well positive
maxSlopeThreshold = 30;

%Variables related to Melt Curve
%Time (s) between images of the microscope tif stack
time_spacing = 30;

%LAMP frames
LAMP_start = 1; %First frame
LAMP_end = 185; %Last frame at 70C (This is used to find the mask)

%MeltCurve frames
MC_start = 194; %First frame at min temp = 54.98C
MC_end = 241; %Last frame at max temp = 95.02C

%Filter selection for the MC Curve
%0 is no filter
%1 filter by threshold
%2 filter by slope
%3 filter by slope and threshold
MC_filter = 1;


%% PROGRAM START HERE
toc

%Generate some variables
baselineStart = frameAtTemp;
baselineEnd = frameAtTemp+5;
LAMP_length = LAMP_end-LAMP_start + 1; 
MC_length = MC_end - MC_start + 1; 
fig_counter = 1;

%Identify all the .tif files
myFiles = dir(fullfile(folder, '*.tif'));

%For each .tif, run image analysis
for a=1:size(myFiles,1)
    %Generate the specific file name
    baseFileName = myFiles(a).name;
    filename = fullfile(folder, baseFileName);
    
    %Load the image into memory
    if reloadIm == 1
        disp(['Loading ', filename])
        tic
        tifStack=ReadImage6D(filename);
        disp(['Loaded ', filename])
        toc
    end
    
    %Make directory for saving data
    %Break the filename into parts
    [filepath, name, ext] = fileparts(filename);

    %Create a folder to save the results
    if imageSave == 1
        mkdir(strcat(saveHere, 'Results_', name));
    end
    
    %The data file tifStack stores image info in {1} and metadata in {2}
    %Grab the metadata
    numImages = tifStack{2}.SizeT;
    sizeX = tifStack{2}.SizeY;
    sizeY = tifStack{2}.SizeX;
    
    %tifStack stores image info {1} in the format below
    %If multi-color tif stack
      %1 = Series
      %2 = SizeT
      %3 = SizeZ
      %4 = SizeC
      %5 = SizeY
      %6 = SizeX
    
    %Load the tempData into memory
    fileID = fopen([filename(1:end-4), '.txt'], 'r');
    formatSpec = ['%s', '%D', '%f', '%f', '%f', '%f', '%f', '%f'];
    tempData = textscan(fileID, formatSpec, 'Delimiter','\t', 'MultipleDelimsAsOne',1, 'HeaderLines', 0);
    timeData = tempData{2};
    timeData = timeData - timeData(1);
    
    %Quick fix for PM to AM in date file
    %If timeData is negative (switched from PM to AM), add 24h
    for d = 1:size(timeData,1)
        if timeData(d) <0
            timeData(d) = timeData(d) + hours(24);
        end
    end
    
    timeData = seconds(timeData);
    probeData = tempData{4};
    setPoint = tempData{6};
    fclose(fileID);
    
    %Align the temperature data with the images
    timeData30s = zeros(1,numImages);
    tempData30s = zeros(1,numImages);
    switcher = 0;

    s = 0;
    for t = 1:size(timeData)
        if timeData(t) > s*time_spacing && s<numImages
           s=s+1;
           timeData30s(s) = timeData(t);
           tempData30s(s) = probeData(t);
        end
    end
    
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    hold on;
    title('Temperature vs. Time')
    ylabel('Temperature (C)')
    xlabel('Time (s)')
    plot(timeData, probeData, 'b');
    plot(timeData, setPoint, 'r');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %FIND ALL WELLS USING ROX
    allMasks = false(MC_end, sizeX, sizeY);
    wellsDetected = zeros(MC_end, size(mask_thresh,2));
    
    %Loop through images and find the best mask for each frame
    disp('Finding Best Masks')
    tic
    for i = 1:MC_end
        
        if mod(i,10)==0
            disp(strcat('Finding best mask #', num2str(i)))
        end
        
        %Detect wells
        bestMask = false(sizeX, sizeY);
        for k = 1:size(mask_thresh,2)
            bwthresh = imbinarize(cast(16*squeeze(tifStack{1}(1,i,1,2,:,:)), 'uint16'), mask_thresh(k));
            %Apply area filter
            areaFilter = bwpropfilt(bwthresh, 'Area', areaBound, 4);
            
            ccTemp = bwconncomp(areaFilter, 4);
            wellsDetected(i,k) = ccTemp.NumObjects;
            
            bestMask = bestMask | areaFilter;
        end
        allMasks(i,:,:) = bestMask;
    end
    disp('Masks Found')
    toc
    
    %imshow(squeeze(allMasks(1,:,:)));
    %imshow(squeeze(allMasks(MC_end,:,:)));
    
    %%
    %Label the first mask
    labeledMask = zeros(MC_end, sizeX, sizeY, 'uint16');
    labeledMask(1,:,:) = uint16(bwlabel(squeeze(allMasks(1,:,:)), 4));
    
    %Loop through images and label remaining masks
    disp('Labeling Masks')
    tic
    for g = 2:MC_end
        if mod(g,10)==0
            disp(strcat('Labeling Mask #', num2str(g)))
        end
        
        thisMask = squeeze(allMasks(g,:,:));
        %Get the center of mass for wells in the previos masks
        prevCC = squeeze(labeledMask(g-1,:,:));
        stats = regionprops(prevCC, 'Centroid');
        centroids = round(cat(1, stats.Centroid));
        centroids(isnan(centroids))=0;
        tempResultMask = zeros(sizeX, sizeY, 'uint16');
        
        %Flood fill labeling, starting at each center-of-mass
        labeler = 1;
        for f = 1:size(centroids, 1)
            
            [particle, thisMask] = floodFill(thisMask, centroids(f,2), centroids(f,1), []);
            particle = reshape(particle, 2, length(particle)/2);
            
            if ~isempty(particle)
                for e = 1:size(particle,2)
                    tempResultMask(particle(1,e), particle(2,e)) = labeler;
                end  
            end
            labeler = labeler+1;
        end
        labeledMask(g, :,:) = tempResultMask;
    end
    disp('Masks Labeled')
    toc
    
    %Get the well Areas
    Frame1_stats = regionprops(squeeze(labeledMask(1,:,:)), 'Area');
    wellAreaFrame1 = cat(1,Frame1_stats.Area);
    
    finalFrame_stats = regionprops(squeeze(labeledMask(end,:,:)), 'Area');
    wellAreaFinalFrame = cat(1,finalFrame_stats.Area);
    wellAreaFinalFrame = wellAreaFinalFrame(wellAreaFinalFrame~=0);
    
    disp('Save tiff and count wells')
    %Save tiff stack of mask detection
    outputFileName = strcat(saveHere, 'Results_', name,'\mask_stack.tif');
    imwrite(squeeze(labeledMask(1, :, :)), outputFileName, 'WriteMode', 'append');
    for K=2:size(labeledMask,1)
       imwrite(squeeze(labeledMask(K, :, :)), outputFileName, 'WriteMode', 'append');
    end
    
    %Count wells
    wellCounts = zeros(1,MC_end);
    
    for c = 1:MC_end
        wellCounts(c)= size(unique(squeeze(labeledMask(c,:,:))),1) - 1;
    end
    
    %Find the unique objects
    wellIdentifiers = unique(labeledMask(MC_end,:,:));
    %Remove wells = to 0 or 1
    wellIdentifiers = wellIdentifiers(2:end);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %FIND INTENSITY OF ALL DETECTED WELLS OVER TIME
    
    %Initialize matrix to track intensity for LAMP and MC
    LAMP_intensity = zeros(LAMP_length, max(wellCounts));
    MC_intensity = zeros(MC_length, max(wellCounts));
    LAMP_intensity_selection = zeros(LAMP_length, min(wellCounts));
    MC_intensity_selection = zeros(MC_length, min(wellCounts));
    
    disp('Calculating Intensities')
    tic
    %Loop through LAMP images and find the avg intensity of each well
    for i = LAMP_start:LAMP_end
        if mod(i,10)==0
            disp(strcat('Calculating image #', num2str(i)))
        end

        current_im_u16 = cast(squeeze(tifStack{1}(1,i,1,1,:,:)), 'uint16');

        LAMP_result = regionprops(squeeze(labeledMask(i,:,:)), current_im_u16, 'MeanIntensity');
        LAMP_intensity(i, 1:size(LAMP_result,1)) = cat(1, LAMP_result.MeanIntensity);
        LAMP_intensity_selection(i, :) = LAMP_intensity(i, wellIdentifiers);
    end
    
    %Loop through Melt Curve images and find the avg intensity of each well
    for i = 1:MC_length
        if mod(i+MC_start-1,10)==0
            disp(strcat('Calculating image #', num2str(i+MC_start-1)))
        end

        current_im_u16 = cast(squeeze(tifStack{1}(1,i+MC_start-1,1,1,:,:)), 'uint16');
        
        MC_result = regionprops(squeeze(labeledMask(i+MC_start-1,:,:)), current_im_u16, 'MeanIntensity');
        MC_intensity(i, 1:size(MC_result,1)) = cat(1, MC_result.MeanIntensity);
        MC_intensity_selection(i, :) = MC_intensity(i, wellIdentifiers);
    end
    disp('Intensities Calculated')
    toc
    
    numWells = min(min(wellCounts));
    
    disp('Begin Analysis')
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %DATA SMOOTHING, BKGND CORRECTION, AND SLOPE CORRECTION
    %Apply heuristically determined moving average window filter
    smooth = smoothdata(LAMP_intensity_selection, 'gaussian', gaussWinSize);

    %Background average subtraction
    baseline = zeros(1, numWells);
    baselineData = LAMP_intensity_selection;

    if baselineAdjust==1
        %Average the intensities between baselineStart and baselineEnd for
        %each well
        for i = 1:numWells
            baseline(i) = mean(smooth(baselineStart:baselineEnd, i));
        end
        
        %Subtract the baseline for each well
        for i = 1:numWells
            for j = 1:LAMP_length
               baselineData(j,i) = smooth(j,i) - baseline(i); 
            end
        end
    end
    
    %Adjust the slope from baseline
    if slopeAdjust==1
        for i = 1:size(baselineData,1)
            slope = (baselineData(baselineEnd, i) - baselineData(baselineStart, i)) / (baselineEnd - baselineStart + 1);
            baselineSlope = linspace(0, (LAMP_length-1)*slope, LAMP_length);
            baselineSlope = baselineSlope - (((baselineEnd+baselineStart)/2)-1)*slope;

            baselineData(i, :) = baselineData(i, :) - baselineSlope(i);
        end
    end
    disp('Finished data smoothing')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %DATA PROCESSING
    
    %Determine which wells cross the intensity threshold
    %Initialize an array to hold time-to-positive values for each well
    ttp_thresh = zeros(numWells,1);

    %Do not count any time-to-positives before sample has reached temp
    for i=frameAtTemp:LAMP_end
        for j=1:numWells
            %Record the first frame that a well crosses the threshold
            if (baselineData(i,j)>threshold && ttp_thresh(j)==0)
                ttp_thresh(j) = i;
            end
        end
    end
    
    %Generate new array that only contains wells that cross the threshold
    ttp_thresh_clean = ttp_thresh(ttp_thresh>0);
    
    %Generate array of number of wells on per frame
    Intensity_Histogram = zeros(LAMP_length,1);
    for i=1:LAMP_length
        Intensity_Histogram(i) = sum(ttp_thresh_clean(:)==i);
    end
    
    %Find the max slope for each well
       fast_Slope = zeros(numWells, 1);
       fast_SlopeFrame_all = zeros(numWells, 1);
       for p = 1:numWells
           [fastestSlope,fastestSlopeFrame] = max(diff(baselineData(:,p)));
           %Ignores extremely high slopes that are imaging artifacts
           if fastestSlope < maxSlope
               fast_Slope(p) = fastestSlope;
               fast_SlopeFrame_all(p) = fastestSlopeFrame;
           end
       end
       fast_Slope_clean=fast_Slope(ttp_thresh>0);
       fast_SlopeFrame_clean=fast_SlopeFrame_all(ttp_thresh>0);
        %To check TTP vs Max Rate Frame
        %ttp_thresh_clean = "Time to Positive"
        %fast_Slope_clean = "Max Rate"
        %fast_SlopeFrame_clean = "Frame Max Rate occurs" 
    
    %Determine which wells cross the intensity AND slope thresholds
    ttp_thresh_slope = ttp_thresh(fast_Slope>maxSlopeThreshold);
    ttp_thresh_slope_clean = ttp_thresh_slope(ttp_thresh_slope>0);
    
    Intensity_Slope_Histogram = zeros(LAMP_length, 1);
    for i=1:LAMP_length
        Intensity_Slope_Histogram(i) = sum(ttp_thresh_slope_clean(:)==i);
    end
    
    %Determine which wells cross slope threshold
    baselineDiffData = diff(baselineData,1,1);
        
    ttp_slopeOnly = zeros(numWells,1);
    for q=frameAtTemp:LAMP_end-1
        for r=1:numWells
            %Record the first frame that a well crosses the threshold
            if (baselineDiffData(q,r)>maxSlopeThreshold && ttp_slopeOnly(r)==0)
                ttp_slopeOnly(r) = q;
            end
        end
    end
    
    %Generate new array that only contains wells that cross the threshold
    ttp_slopeOnly_clean = ttp_slopeOnly(ttp_slopeOnly>0);
    
    Slope_Histogram = zeros(LAMP_length, 1);
    for i=1:LAMP_length
        Slope_Histogram(i) = sum(ttp_slopeOnly_clean(:)==i);
    end
    
    %Find the max intensity of each positive well
    endIntensity = zeros(numWells, 1);
    for m = 1:numWells
        endIntensity(m) = (baselineData(LAMP_length,m));
    end
    
    %{
    Intensity20min = zeros(numWells, 1);
    for m = 1:numWells
        Intensity20min(m) = (baselineData(71,m));
    end
    whatWeWant = Intensity20min(ttp_thresh>0);
    %}
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %MC DATA PROCESSING
    %Filter Melt Curve by variable setting
    %threshold filter
    filtered_Slope = fast_Slope;
    if MC_filter==1
        MC_intensity = MC_intensity_selection(:, ttp_thresh>0);
        filtered_Slope = fast_Slope(ttp_thresh>0);
    %slope filter
    elseif MC_filter ==2
        MC_intensity = MC_intensity_selection(:, ttp_slopeOnly>0);
        filtered_Slope = fast_Slope(ttp_thresh>0);
    %slope thresh filter
    elseif MC_filter ==3
        MC_intensity = MC_intensity_selection(:, ttp_thresh_slope>0);
        filtered_Slope = fast_Slope(ttp_thresh>0);
    end
    
    numPos = size(MC_intensity, 2);
    
    %Apply heuristically determined moving average window filter
    smooth_MC = smoothdata(MC_intensity, 'gaussian', gaussWinSize);
    
    %Determine melt curve derivative
    MC_DerivData = -diff(smooth_MC,1,1);
    
    MC_PeakMeltTemps = zeros(numPos,1);
    %Calculate the peak temp for each curve
    for u=1:numPos
       [temp, max_idx] = max(MC_DerivData(:,u));
       MC_PeakMeltTemps(u)=tempData30s(MC_start+max_idx);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Plot the generation of the masks
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    subplot(2,2,1);
    imshow(cast(16*squeeze(tifStack{1}(1,1,1,2,:,:)), 'uint16'));
    title('First Image')
    
    subplot(2,2,2);
    imshow(cast(16*squeeze(tifStack{1}(1,end,1,2,:,:)), 'uint16'));
    title('Last Image')
    
    subplot(2,2,3);
    imshow(squeeze(labeledMask(1,:,:)));
    title('First Mask')
    
    subplot(2,2,4);
    imshow(squeeze(labeledMask(end,:,:)))
    title('Final Mask')
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\Thresholding'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot raw LAMP intensities
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    %Downsample here to shown fewer curves
    plot(timeData30s(LAMP_start:LAMP_end), LAMP_intensity_selection(:, :)); 
    
    %Set the plot axes
    title([name ' raw LAMP Intensity']);
    xlabel('Time (s)');
    ylabel('RFU');
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\RawIntensity'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot baselined intensities over time
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    %Downsample here to shown fewer curves
    plot(timeData30s(LAMP_start:LAMP_end), baselineData(:, :)); 
    
    %Set the plot axes
    title([name ' Corrected Well Intensity']);
    xlabel('Time (s)');
    ylabel('RFU');
    refline(0, threshold);
     
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\BaselineIntensity'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot derivative of baseline over time
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    %Downsample here to shown fewer curves
    plot(timeData30s(LAMP_start:LAMP_end-1), baselineDiffData(:, :)); 
    
    %Set the plot axes
    title([name ' Corrected Derivative']);
    xlabel('Time (s)');
    ylabel('Delta RFU');
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\BaselineDeriv'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot MC intensities
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    %Downsample here to shown fewer curves
    plot(tempData30s(MC_start:MC_end), MC_intensity_selection(:, :)); 
    
    %Set the plot axes
    title([name ' raw MC Intensity']);
    xlabel('Temp (C)');
    ylabel('RFU');
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\MCIntensity'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot MC derivative
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    %Downsample here to shown fewer curves
    plot(tempData30s(MC_start:MC_end-1), MC_DerivData(:, :)); 
    
    %Set the plot axes
    title([name ' MC Derivative']);
    xlabel('Temp (C)');
    ylabel('RFU');
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\MCIntensity'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot MC Histogram
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    histogram(MC_PeakMeltTemps);
    %Set the plot axes
    title([name ' MC PeakMeltTemps']);
    xlabel('Temp (C)');
    ylabel('Counts');
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\MCPeakWellsTemps'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Plot MC peak melt vs. max Slope
    figure(fig_counter);
    fig_counter = fig_counter + 1;
    
    scatter(filtered_Slope, MC_PeakMeltTemps)
    
    %Set the plot axes
    title([name ' MC_PeakMelt vs. maxSlope']);
    xlabel('maxSlope');
    ylabel('MeltTemp');
    
    if imageSave == 1
        saveas(gcf,strcat(saveHere, 'Results_', name,'\Melt_v_slope'),'png')
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%PLOT HISTOGRAM AND CDF FOR WELLS THAT CROSS INTENSITY THRESH
    
    %Only plot if the are wells that turned positive. 
    if ttp_thresh_clean > 1    
        figure(fig_counter);
        fig_counter = fig_counter + 1;
        h=histcounts(ttp_thresh_clean, max(max(ttp_thresh_clean)) - min(min(ttp_thresh_clean)) + 1);

        %LineSpec Settings: line, marker, and colour
        plot(min(min(ttp_thresh_clean)):max(max(ttp_thresh_clean)), h, 'b');
        axis([0 LAMP_length -inf inf]);
        title([name ' Intensity Histogram'])
        ylabel('Wells Turning On')
        xlabel('Frame Number (2 per Minute)')
        if imageSave == 1
            saveas(gcf,strcat(saveHere, 'Results_', name,'\Intensity_Hist'),'png')
        end
        
        %%Plot ttp CDF
        figure(fig_counter);
        fig_counter = fig_counter + 1;
        intensity_cdfhandle = cdfplot(ttp_thresh_clean);  
        title([name ' Intensity CDF, ', num2str(size(ttp_thresh_clean,1)), ' Wells'])
        xlabel('Frame Number (2 per Minute)')
        set(intensity_cdfhandle, 'LineStyle', '-', 'Color', 'r');
        
        if imageSave == 1
            saveas(gcf,strcat(saveHere, 'Results_', name,'\Intensity_CDF'),'png')
        end
        
        %%
        %Move this section under the plots that you want it to calculate
        %the statistics for. 
        %When interpolating, assumes monotonic and linear between points on
        %the histogram. 
        
        %Calculate Graph Statistics
        [max_height, max_ind] = max(h);
        fiftyPercentMax = max_height*.5;
        tenPercentMax = max_height*.1;
        fivePercentMax = max_height*.05;
        
        %Calculate left fiftyPercent max time
        b=max_ind;
        while h(b)>fiftyPercentMax &&b~=1
            b=b-1;
        end
        leftFiftyPercentMaxInd = b + ((fiftyPercentMax - h(b)) / (h(b+1)-h(b)));
        
        %Calculate right fiftyPercent max time
        c=max_ind;
        while h(c)>fiftyPercentMax&&c~=size(h,2)
            c=c+1;
        end
        rightFiftyPercentMaxInd = c-1 + ((h(c-1) - fiftyPercentMax) / (h(c-1)-h(c)));
        
        %Calculate left tenPercent max time
        d=max_ind;
        while h(d)>tenPercentMax &&d~=1
            d=d-1;
        end
        leftTenPercentMaxInd = d + ((tenPercentMax - h(d)) / (h(d+1)-h(d)));
        
        %Calculate right tenPercent max time
        e=max_ind;
        while h(e)>tenPercentMax&&e~=size(h,2)
            e=e+1;
        end
        rightTenPercentMaxInd = e-1 + ((h(e-1)-tenPercentMax) / (h(e-1)-h(e)));
        
        %Calculate left fivePercent max time
        f=max_ind;
        while h(f)>fivePercentMax &&f~=1
            f=f-1;
        end
        leftFivePercentMaxInd = f + ((fivePercentMax - h(f)) / (h(f+1)-h(f)));
        
        %Calculate right fivePercent max time
        g=max_ind;
        while h(g)>fivePercentMax&&g~=size(h,2)
            g=g+1;
        end
        rightFivePercentMaxInd = g-1 + ((h(g-1)-fivePercentMax) / (h(g-1)-h(g)));
        
        %Calculate the important metrics
        max_time = max_ind + min(min(ttp_thresh_clean));
        halfMaxTime = rightFiftyPercentMaxInd - leftFiftyPercentMaxInd;
        asymmetryFactor = (rightTenPercentMaxInd-max_ind) / (max_ind-leftTenPercentMaxInd);
        tailingFactor = (max_ind-leftFivePercentMaxInd + rightFivePercentMaxInd-max_ind)/ 2*(max_ind-leftFivePercentMaxInd);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%PLOT HISTOGRAM AND CDF FOR WELLS THAT CROSS INTENSITY AND SLOPE THRESH
    if ttp_thresh_slope_clean > 1    
        figure(fig_counter);
        fig_counter = fig_counter + 1;
        h=histcounts(ttp_thresh_slope_clean, max(max(ttp_thresh_slope_clean)) - min(min(ttp_thresh_slope_clean)) + 1);

        %LineSpec Settings: line, marker, and colour
        plot(min(min(ttp_thresh_slope_clean)):max(max(ttp_thresh_slope_clean)), h, 'b');
        axis([0 LAMP_length -inf inf]);
        title([name 'Intensity Slope Histogram'])
        ylabel('Wells Turning On')
        xlabel('Frame Number (2 per Minute)')
        if imageSave == 1
            saveas(gcf,strcat(saveHere, 'Results_', name,'\Intensity_Slope_Hist'),'png')
        end
        
        figure(fig_counter);
        fig_counter = fig_counter + 1;
        intensity_slope_cdfhandle = cdfplot(ttp_thresh_slope_clean);  
        title([name ' Intensity Slope CDF, ', num2str(size(ttp_thresh_slope_clean,1)), ' Wells'])
        xlabel('Frame Number (2 per Minute)')
        set(intensity_slope_cdfhandle, 'LineStyle', '-', 'Color', 'r');

        if imageSave == 1
            saveas(gcf,strcat(saveHere, 'Results_', name,'\Intensity_Slope_CDF'),'png')
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%PLOT HISTOGRAM AND CDF FOR WELLS THAT SLOPE ONLY THRESH
    if ttp_slopeOnly_clean > 1    
        figure(fig_counter);
        fig_counter = fig_counter + 1;
        h=histcounts(ttp_slopeOnly_clean, max(max(ttp_slopeOnly_clean)) - min(min(ttp_slopeOnly_clean)) + 1);

        %LineSpec Settings: line, marker, and colour
        plot(min(min(ttp_slopeOnly_clean)):max(max(ttp_slopeOnly_clean)), h, 'b');
        axis([0 LAMP_length -inf inf]);
        title([name 'Intensity Slope Histogram'])
        ylabel('Wells Turning On')
        xlabel('Frame Number (2 per Minute)')
        if imageSave == 1
            saveas(gcf,strcat(saveHere, 'Results_', name,'\Slope_Hist'),'png')
        end
        
        figure(fig_counter);
        fig_counter = fig_counter + 1;
        slope_cdfhandle = cdfplot(ttp_slopeOnly_clean);  
        title([name ' Intensity Slope CDF, ', num2str(size(ttp_slopeOnly_clean,1)), ' Wells'])
        xlabel('Frame Number (2 per Minute)')
        set(slope_cdfhandle, 'LineStyle', '-', 'Color', 'r');

        if imageSave == 1
            saveas(gcf,strcat(saveHere, 'Results_', name,'\Slope_CDF'),'png')
        end
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %SAVE TO EXCEL
    
    %SUMMARY FILE
    %TTP_hist: Contains the number of wells that turned on for each frame
    %fast_TTP_hist: Contains the number of wells that turned on after
    %   filtering by slope threshold
    %ttp: Lists time-to-positive for each tracked well that
    %   crosses the treshold
    %fast_ttp: Lists time-to-positive for each tracked well that
    %   crosses the treshold and slope threshold
    %maxSlope: Maximum slope/frame for each well
    %endIntensity: Final intensity (after baseline) for each well

    %Total wells: Total wells detected on frame 1
    %Detected wells: number of wells detected on 2nd to last frame
    %Positive wells: number of detected wells that cross threshold
    %Fast slope pos wells: number of positive wells that cross slope threshold
    excelEntries = numWells;
    if MC_end > excelEntries
        excelEntries = MC_end;
    end
    if numImages > excelEntries
        excelEntries=numImages;
    end
    
    col_header={'Intensity_well#', 'Intensity_well_ttp','Intensity_well_PeakMeltTemps', 'Intensity_well_maxSlope', 'Intensity_endIntensity', 'Intensity_well_hist', ...
        'Intensity_slope_ttp', 'Intensity_slope_hist', 'slope_ttp', 'slope_hist','maxSlope_all', 'endIntensity_all', 'WellCounts', ...
        'Area_finalFrame', 'Temp30s','', '', 'mode TTP', 'FWHM', 'Asymmetric','Tailing'};
    col0 = cell(excelEntries, 1); col0(1:size(find(ttp_thresh>0),1),1) = num2cell(find(ttp_thresh>0));
    col1 = cell(excelEntries, 1); col1(1:size(ttp_thresh_clean,1),1) = num2cell(ttp_thresh_clean);
    col2 = cell(excelEntries, 1); col2(1:size(MC_PeakMeltTemps,1),1) = num2cell(MC_PeakMeltTemps);
    colMaxSlopeThresh = cell(excelEntries, 1); colMaxSlopeThresh(1:size(fast_Slope(ttp_thresh>0),1),1) = num2cell(fast_Slope(ttp_thresh>0));
    colEndIntensityThresh = cell(excelEntries, 1); colEndIntensityThresh(1:size(endIntensity(ttp_thresh>0),1),1) = num2cell(endIntensity(ttp_thresh>0));
    col3 = cell(excelEntries, 1); col3(1:size(Intensity_Histogram,1),1) = num2cell(Intensity_Histogram);
    col4 = cell(excelEntries, 1); col4(1:size(ttp_thresh_slope_clean,1),1) = num2cell(ttp_thresh_slope_clean);
    col5 = cell(excelEntries, 1); col5(1:size(Intensity_Slope_Histogram,1),1) = num2cell(Intensity_Slope_Histogram);
    col6 = cell(excelEntries, 1); col6(1:size(ttp_slopeOnly_clean,1),1) = num2cell(ttp_slopeOnly_clean);
    col7 = cell(excelEntries, 1); col7(1:size(Slope_Histogram,1),1) = num2cell(Slope_Histogram);
    
    col8 = cell(excelEntries, 1); col8(1:size(fast_Slope,1),1) = num2cell(fast_Slope);
    col9 = cell(excelEntries, 1); col9(1:size(endIntensity,1),1) = num2cell(endIntensity);
    colAreaFinalFrame = cell(excelEntries, 1); colAreaFinalFrame(1:size(wellAreaFinalFrame,1),1) = num2cell(wellAreaFinalFrame);
    col10 = cell(excelEntries, 1); col10(1:size(wellCounts,2),1) = num2cell(wellCounts);
    col11 = cell(excelEntries, 1); col11(1:size(tempData30s,2),1) = num2cell(tempData30s);
    col12= cell(excelEntries, 1); col12(1:19,1) = {'Intensity wells'; 'Intensity slope wells'; 'slope wells';...
        ''; 'Settings'; 'frameAtTemp'; 'baselineStart'; 'baselineEnd'; 'areaLowBound'; 'areaHighBound';...
        'threshold'; 'GaussWindow'; 'maxSlope'; 'maxSlopeThreshold'; 'LAMP_start'; 'LAMP_end'; 'MC_start'; 'MC_end';...
        'MC_filter'};
    col13 = cell(excelEntries, 1); col13(1:19,1) = num2cell([size(ttp_thresh_clean,1); size(ttp_thresh_slope_clean,1); size(ttp_slopeOnly_clean,1);...
        0; 0; frameAtTemp; baselineStart; baselineEnd; areaBound(1); areaBound(2); threshold; gaussWinSize; ...
        maxSlope; maxSlopeThreshold; LAMP_start; LAMP_end; MC_start; MC_end; MC_filter]);
    col13(4:5,1)={''; ''};
    col14 = cell(excelEntries, 1);col14(1,1) = num2cell(max_time);
    col15 = cell(excelEntries, 1);col15(1,1) = num2cell(halfMaxTime);
    col16 = cell(excelEntries, 1);col16(1,1) = num2cell(asymmetryFactor);
    col17 = cell(excelEntries, 1);col17(1,1) = num2cell(tailingFactor);

    saveSummary = [col_header; col0 col1 col2 colMaxSlopeThresh colEndIntensityThresh col3 col4 col5 col6 col7 col8 col9 ...
        colAreaFinalFrame col10 col11 col12 col13 col14 col15 col16 col17];

    if excelSaveSummary == 1
        disp('Saving Summary Excel File')
        xlswrite(strcat(saveHere, 'Results_', name,'\',name,' Summary.xlsx'), saveSummary)
    end 

    %INTENSITY FILE
    %Total number of wells stored as the name of the sheet in Intensity
    %Intensity: raw traces for each tracked well
    %baselineData: traces after gaussian smoothing, baseline avg
    %subtraction and slope correction (if applicable).
    if excelSaveIntensity == 1
        disp('Saving Intensity Excel File')
        xlswrite(strcat(saveHere, 'Results_', name,'\',name,' Intensity.xlsx'), transpose(LAMP_intensity_selection), strcat('Intensity'))
        xlswrite(strcat(saveHere, 'Results_', name,'\',name,' Intensity.xlsx'), transpose(baselineData), 'BaselineCorrected')
        xlswrite(strcat(saveHere, 'Results_', name,'\',name,' Intensity.xlsx'), transpose(MC_intensity_selection), 'MC_Intensity')
        xlswrite(strcat(saveHere, 'Results_', name,'\',name,' Intensity.xlsx'), transpose(MC_intensity), 'MC_Intensity_PosOnly')
        xlswrite(strcat(saveHere, 'Results_', name,'\',name,' Intensity.xlsx'), transpose(MC_DerivData), 'MC_Deriv')
    end 
    
    %Plot 4d Video
    disp('Plot 4D')
    %Final Intensity coloring
        color=endIntensity(ttp_thresh>0);
        colormap(jet);
        scatter3(ttp_thresh_clean,MC_PeakMeltTemps,fast_Slope(ttp_thresh>0),5,color,'filled');
    %limit axis sizes for video (X_min X_Max Y_min Y_Max Z_Min Z_Max)
        axis([0 max_time 60 100 0 maxSlope]);
        xlabel('TTP in Frames');
        ylabel('Melting Temperature in °C');
        zlabel('MaxRate');
        title(name);
        c=colorbar;
        c.Label.String = 'Final Intensity';      
      
    %Video Rotate
        OptionZ.FrameRate=20;OptionZ.Duration=20;OptionZ.Periodic=true;
        CaptureFigVid([-20,10;-110,0;-180,90;-290,10;-380,10], strcat(saveHere, 'Results_', name,'\',name,' 4D Plot'), OptionZ);

        if imageSave == 1;
            savefig(strcat(saveHere, 'Results_', name,'\4dPlot'));
        end
        
    disp('Finish analysis')
    toc
    
    if saveVars == 1
        save(strcat(saveHere, 'Results_', name,'\',name,' Variables.mat'));
    end
end