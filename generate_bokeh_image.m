function generate_bokeh_image()

%% Generate bokeh effect from a single image
close all;

imgPath = '34.jpg';

blurSigma = 1;
dBlurThresh = 3;
upSample_type = 'linear';

%% Load Image
img_orig = imread(imgPath);

%% Resize Image
[~, nC, ~] = size(img_orig);
img_scale = min(1, 640/nC);
img_orig = imresize(img_orig, img_scale);
%% Process Image
img = rgb2gray(im2double(img_orig));

h1 = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
subplot(3, 3, 1); imshow(img_orig, []); title('Original Image');
subplot(3, 3, 2); imshow(img, []); title('Gray-scale Image');

%% Get blurred edges
ix = img;   % Get original image.

% Get re-blurred image
blur_kernel = fspecial('gaussian', [4*blurSigma+1 4*blurSigma+1], blurSigma);
ixx = imfilter(img, blur_kernel, 'replicate');

figure(h1);
subplot(3, 3, 3); imshow(ixx, []); title('Blurred Image');
figure();
subplot(1, 2, 1); imshow(ix, []); title('Original Image');
subplot(1, 2, 2); imshow(ixx, []); title('Blurred Image');

% Get gradient of original image
[gx_ix, gy_ix] = imgradientxy(ix);
gm_ix = (gx_ix.^2+gy_ix.^2).^0.5;

% Get gradient of re-blurred image
[gx_ixx, gy_ixx] = imgradientxy(ixx);
gm_ixx = (gx_ixx.^2+gy_ixx.^2).^0.5;

figure(h1);
subplot(3, 3, 4); imshow(gm_ix, []); title('Gradient Original');
subplot(3, 3, 5); imshow(gm_ixx, []); title('Gradient Blurred');
figure();
subplot(1, 2, 1); imshow(gm_ix, []); title('Gradient Original');
subplot(1, 2, 2); imshow(gm_ixx, []); title('Graident Blurred');

% Get gradient ratio 
R = gm_ix./gm_ixx;
[xl, yl] = meshgrid(1:size(R, 2), 1:size(R, 1));
yl = flipud(yl);

% Get defocus blur
maskR = (R<=1);
DB = blurSigma./sqrt(R.^2-1);
DB(maskR) = 0;
DB = medfilt2(DB, [3 3]);
DB(DB>dBlurThresh) = dBlurThresh;

figure(h1);
subplot(3, 3, 6); imshow(R, []); title('R');
subplot(3, 3, 7); imshow(DB, []); title('DB');
figure();
subplot(1, 2, 1); imshow(R, []); title('R');
subplot(1, 2, 2); imshow(DB, []); title('DB');

% Get edges
[~,threshOut] = edge(ix, 'canny');
sigma = sqrt(2);
thresh = [threshOut(2)*0.85, threshOut(2)];
[bw] = edge(ix, 'canny', thresh, sigma);

% Get defocus blur only along the edges
DB_edge = DB.*bw;
DB_edge(DB_edge == 0) = NaN;

% Remove outliers
DBtemp = sort(DB_edge(~isnan(DB_edge)));
thresh_DBedge = mean(DBtemp) + 6*std(DBtemp);
DB_edge(DB_edge>thresh_DBedge) = thresh_DBedge;

% Normalize to [0 1]
DB_edge_norm = DB_edge - min(DB_edge(:));
DB_edge_norm = DB_edge_norm ./ max(DB_edge_norm(:));

figure(h1);
subplot(3, 3, 8); imshow(bw, []); title('Edges');
subplot(3, 3, 9); imshow(DB_edge, []); title('DB-edges');
h2 = figure('units', 'normalized', 'outerposition', [0 0 1 1]);
subplot(1, 2, 1); imshow(DB_edge_norm, []); title('Sparse Defocus Map');

%% Apply bilateral filtering
sigmab = [5 0.3];

W = round(sigmab(1))*4 + 1;
DB_edge_BF = bilateral_filter(DB_edge_norm, W, sigmab);

figure(h2);
subplot(1, 2, 2); imshow(DB_edge_BF, []); title('Sparse Defocus Map with Bilateral Filter');

%% Get Matting-Laplacian matrix
[height, width] = size(bw);
L = get_laplacian_matrix(im2double(img_orig), 1);

%% Get Full Defocus map
lambda = 0.005;

bwReshape = reshape(bw, [height*width, 1]);
DBReshape = reshape(DB_edge_BF, [height*width,1]);
DBReshape(isnan(DBReshape)) = 0;

D = spdiags(bwReshape(:), 0, height*width, height*width);
DBss = (L+lambda*D)\(lambda*D*DBReshape);

DBss = reshape(DBss, [height, width]);

F_DB = scatteredInterpolant(xl(:), yl(:), DBss(:), upSample_type);
DB_fullMP = F_DB(xl, yl);

figure();
subplot(1, 1, 1); imshow(DB_fullMP, []); title('Full Defocus Map');

%% Post processing
step_count = 3; 
step_interval = linspace(min(DB_fullMP(:)), max(DB_fullMP(:)), step_count);
blur_sigma = [2 5 10];
bokeh_img = uint8(zeros(size(img_orig)));
for i = 1:step_count-1
   blur_mask = DB_fullMP>=step_interval(i) & DB_fullMP<step_interval(i+1);
   if i == 1
    blur_section_R = img_orig(:, :, 1).*uint8(blur_mask);
    blur_section_G = img_orig(:, :, 2).*uint8(blur_mask);
    blur_section_B = img_orig(:, :, 3).*uint8(blur_mask);
    blur_section(:, :, 1) = blur_section_R;
    blur_section(:, :, 2) = blur_section_G;
    blur_section(:, :, 3) = blur_section_B;
    section_after_blur = blur_section;
    bokeh_img = section_after_blur + bokeh_img;
   else
    kernel_blur = fspecial('gaussian', [4*blur_sigma(i)+1 4*blur_sigma(i)+1], blur_sigma(i));
    orig_img_after_blur = imfilter(img_orig, kernel_blur, 'replicate');
    
    blur_section_R = orig_img_after_blur(:, :, 1).*uint8(blur_mask);
    blur_section_G = orig_img_after_blur(:, :, 2).*uint8(blur_mask);
    blur_section_B = orig_img_after_blur(:, :, 3).*uint8(blur_mask);
    blur_section(:, :, 1) = blur_section_R;
    blur_section(:, :, 2) = blur_section_G;
    blur_section(:, :, 3) = blur_section_B;
    section_after_blur = blur_section;

    bokeh_img = section_after_blur + bokeh_img;
   end   
end
    figure();
    subplot(1, 2, 1); imshow(img_orig); title('Original test image');
    subplot(1, 2, 2); imshow(uint8(bokeh_img)); title('Bokeh image');

%% Face detection to avoid false detection of foreground
FaceDetector = vision.CascadeObjectDetector;
BBOX = step(FaceDetector, img_orig);
figure(); imshow(img_orig); hold on
for i = 1:size(BBOX, 1)
    rectangle('Position', BBOX(i,:), 'LineWidth', 5, 'LineStyle', '-', 'EdgeColor', 'r')
end
title('Face Detection');
hold off;

figure(); imshow(img_orig); hold on
scale_factor = 1.3;
BBOX(1,4)= BBOX(1,4)*scale_factor;
BBOX(1,2)= BBOX(1,2)-BBOX(1,4)*(scale_factor-1)*0.8;
for i = 1:size(BBOX,1)
    rectangle('Position', BBOX(i,:), 'LineWidth', 5, 'LineStyle', '-', 'EdgeColor', 'r')
end
title('Adjusted Face Detection');
hold off;

face_mask = zeros([size(img_orig, 1), size(img_orig, 2)]);
face_mask(BBOX(1, 2):(BBOX(1, 4) + BBOX(1, 2)), BBOX(1, 1):(BBOX(1, 3) + BBOX(1,1))) = 1;

face_section = img_orig.*uint8(face_mask);
inverse_face_mask = ones([size(img_orig, 1), size(img_orig, 2)]);
inverse_face_mask(BBOX(1, 2):(BBOX(1, 4) + BBOX(1, 2)), BBOX(1, 1):(BBOX(1, 3) + BBOX(1, 1))) = 0;
bokeh_img_face = face_section + bokeh_img.*uint8(inverse_face_mask);
figure();
subplot(1,3,1); imshow(img_orig); title('Original test image');
subplot(1,3,2); imshow(uint8(bokeh_img)); title('Bokeh image');
subplot(1,3,3); imshow(uint8(bokeh_img_face)); title('Bokeh image with face protection');