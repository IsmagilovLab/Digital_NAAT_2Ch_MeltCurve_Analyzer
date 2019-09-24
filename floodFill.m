function [particle, frame] = floodFill(frame, curRow, curCol, particle)
%Do not check out of frame
if curRow ==0 || curRow == size(frame,1)+1 || curCol == 0 || curCol == size(frame,2)+1
    return
end

%Quit if current position is off
if ~frame(curRow, curCol)
    return
end

%Add the current position to the particle
particle = [particle, curRow, curCol];

%Set the current position off to prevent infinite
frame(curRow,curCol) = 0;

[particle, frame] = floodFill(frame, curRow+1, curCol, particle);
[particle, frame] = floodFill(frame, curRow-1, curCol, particle);
[particle, frame] = floodFill(frame, curRow, curCol+1, particle);
[particle, frame] = floodFill(frame, curRow, curCol-1, particle);

