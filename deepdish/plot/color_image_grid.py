from __future__ import division, print_function, absolute_import
import numpy as np
import deepdish as dd

# If skimage is available, the image returned will be wrapped
# in the Image class. This is nice since it will be automatically
# displayed in an IPython notebook.
def _load_image_class():
    try:
        from skimage.io import Image
    except ImportError:
        def Image(x):
            return x
    return Image


# TODO: ImageGrid and ColorImageGrid need to be integrated more. Since both
# represent the data as an RGB image, ColorImageGrid should be a suitable
# base class.
class ColorImageGrid:
    """
    An image grid used for combining equally-sized RGB images into a
    single larger image. It is similar to `ImageGrid`, except it only
    works for RGB images with color channels scaled in [0, 1].

    Parameters
    ----------
    data : ndarray, ndim in [3, 4, 5]
        The last three axes should be spatial dimensions and a color channel.
        The rest are used to index them. If `ndim` is 3, then a single image is
        shown. If `ndim` is 4, then `rows` and `cols` will determine its
        layout. If `ndim` is 5, then `rows` and `cols` will be ignored and the
        grid will be layed out according to its first two axes instead.  If
        data is set to None, then an empty image grid will be initialized. In
        this case, rows and cols are both required.
    rows/cols : int or None
        The number of rows and columns for the grid. If both are None, the
        minimal square grid that holds all images will be used. If one is
        specified, the other will adapt to hold all images. If both are
        specified, then it possible that the grid will be vacuous or that
        some images will be omitted.
    shape : tuple
        Shape of the each grid image. Only use if ``data`` is set to `None`,
        since it should otherwise be inferred from the data.
    border_color : float or np.ndarray of length 3
        Specify the border color as an array of length 3 (RGB). If a scalar is
        given, it will be interpreted as the grayscale value.
    border_width :
        Border with in pixels. If you rescale the image, the border will be
        rescaled with it.
    cmap/vmin/vmax/vsym :
        See `ImageGrid.set_image`.
    global_bounds : bool
        If this is set to True and either `vmin` or `vmax` is not
        specified, it will infer it globally for the data. If `vsym` is
        True, the global bounds will be symmetric around zero. If it is set
        to False, it determines range per image, which would be the
        equivalent of calling `set_image` manually with `vmin`, `vmax` and
        `vsym` set the same.
    """
    def __init__(self, data=None, rows=None, cols=None, shape=None,
                 border_color=1, border_width=1, vmin=0.0,
                 vmax=1.0, vsym=False, global_bounds=True):

        assert data is None or (np.ndim(data) in (3, 4, 5) and
                                data.shape[-1] <= 3)
        if data is not None:
            data = np.asanyarray(data)

        if data is None:
            assert rows is not None and cols is not None, \
                "Must specify rows and cols if no data is specified"
            shape = shape

        elif data.ndim == 3:
            N = 1
            rows = 1
            cols = 1
            data = data[np.newaxis]
            shape = data.shape[1:3]

        elif data.ndim == 4:
            N = data.shape[0]

            if rows is None and cols is None:
                cols = int(np.ceil(np.sqrt(N)))
                rows = int(np.ceil(N / cols))
            elif rows is None:
                rows = int(np.ceil(N / cols))
            elif cols is None:
                cols = int(np.ceil(N / rows))
            shape = data.shape[1:3]

        elif data.ndim == 5:
            assert rows is None and cols is None
            rows = data.shape[0]
            cols = data.shape[1]
            data = data.reshape((-1,) + data.shape[2:])
            N = data.shape[0]
            shape = data.shape[1:3]

        self._border_color = self._prepare_color(border_color)
        self._rows = rows
        self._cols = cols
        self._shape = shape
        self._border = border_width

        b = self._border
        self._fullsize = (b + (shape[0] + b) * self._rows,
                          b + (shape[1] + b) * self._cols)

        self._data = np.ones(self._fullsize + (3,), dtype=np.float64)

        if global_bounds:
            if vmin is None:
                vmin = np.nanmin(data)
            if vmax is None:
                vmax = np.nanmax(data)

            if vsym:
                mx = max(abs(vmin), abs(vmax))
                vmin = -mx
                vmax = mx

        # Populate with data
        for i in range(min(N, rows * cols)):
            self.set_image(data[i], i // cols, i % cols,
                           vmin=vmin, vmax=vmax, vsym=vsym)

    @classmethod
    def _prepare_color(self, color):
        if color is None:
            return np.array([1.0, 1.0, 1.0])
        elif isinstance(color, (int, float)):
            return np.array([color]*3)
        else:
            return np.array(color)

    @property
    def image(self):
        """
        Returns the image as a skimage.io.Image class.
        """
        Image = _load_image_class()
        return Image(self._data)

    def set_image(self, image, row, col, vmin=0.0, vmax=1.0, vsym=False):
        """
        Sets the data for a single window.

        Parameters
        ----------
        image : ndarray, ndim=3
            The shape should be the same as the `shape` specified when
            constructing the image grid, plus an axis of length 3 for
            the color channels.
        row/col : int
            The zero-index of the row and column to set.
        vmin/vmax : numerical or None
            Defines the range of the color palette. None, takes the range of
            the data. Default is vmin=0 and vmax=1, for plotting normal RGB
            images.
        vsym : bool
            If True, this means that the color palette will always be centered
            around 0. Even if you have specified both `vmin` and `vmax`, this
            will override that and extend the shorter one. Good practice is to
            specify neither `vmin` or `vmax` or only `vmax` together with this
            option.
        """
        if vmin is None:
            vmin = np.nanmin(image)
        if vmax is None:
            vmax = np.nanmax(image)

        if vsym and -vmin != vmax:
            mx = max(abs(vmin), abs(vmax))
            vmin = -mx
            vmax = mx

        if vmin == vmax:
            diff = 1
        else:
            diff = vmax - vmin

        img_scaled = np.clip((image - vmin) / diff, 0, 1)

        x0 = row * (self._shape[0] + self._border)
        x1 = (row + 1) * (self._shape[0] + self._border) + self._border
        y0 = col * (self._shape[1] + self._border)
        y1 = (col + 1) * (self._shape[1] + self._border) + self._border

        self._data[x0:x1, y0:y1] = self._border_color

        anchor = (self._border + row * (self._shape[0] + self._border),
                  self._border + col * (self._shape[1] + self._border))

        selection = [slice(anchor[0], anchor[0] + image.shape[0]),
                     slice(anchor[1], anchor[1] + image.shape[1])]

        C = image.shape[-1]
        if C == 1:
            # Grayscale image
            self._data[selection] = img_scaled
        elif C == 2:
            # If two color channels, use only the first two and fill the third
            # with the average of the two.
            # Create a red-cyan 3D image representation from the two color
            # channels
            l = [img_scaled, img_scaled[..., [1]]]
            nimage = np.concatenate(l, axis=-1)
            self._data[selection] = nimage
        elif C == 3:
            self._data[selection] = img_scaled

    def highlight(self, col=None, row=None, color=None):
        # TODO: This function is not done yet and needs more work

        bw = self._border
        M = np.ones(tuple(np.add(self._shape, 2 * bw)) + (1,), dtype=np.bool)
        M[bw:-bw, bw:-bw] = 0

        def setup_axis(axis, count):
            if axis is None:
                return list(range(count))
            elif isinstance(axis, int):
                return [axis]
            else:
                return axis

        # TODO: This is temporary
        cols = [col] * self._rows
        rows = list(range(self._rows))

        color = self._prepare_color(color)
        for c, r in zip(cols, rows):
            r0 = (self._border + self._shape[0]) * r
            c0 = (self._border + self._shape[1]) * c

            sel = [slice(r0, r0+M.shape[0]), slice(c0, c0+M.shape[1])]

            self._data[sel] = M * color + ~M * self._data[sel]

    def scaled_image(self, scale=1):
        """
        Returns a nearest-neighbor upscaled scaled version of the image.

        Parameters
        ----------
        scale : int
            Upscaling using nearest neighbor, e.g. a scale of 5 will make each
            pixel a 5x5 rectangle in the output.

        Returns
        -------
        scaled_image : skimage.io.Image, (height, width, 3)
            Returns a scaled up RGB image. If you do not have scikit-image, it
            will be returned as a regular Numpy array. The benefit of wrapping
            it in `Image`, is so that it will be automatically displayed in
            IPython notebook, without having to issue any drawing calls.
        """
        if scale == 1:
            return self._data
        else:
            from skimage.transform import resize
            data = resize(self._data, tuple([self._data.shape[i] * scale
                                             for i in range(2)]), order=0)
            Image = _load_image_class()
            return Image(data)

    def save(self, path, scale=1):
        """
        Save the image to file.

        Parameters
        ----------
        path : str
            Output path.
        scale : int
            Upscaling using nearest neighbor, e.g. a scale of 5 will make each
            pixel a 5x5 rectangle in the output.
        """
        data = self.scaled_image(scale)
        dd.image.save(path, data)

    def __repr__(self):
        return ('ColorImageGrid(rows={rows}, cols={cols}, shape={shape})'.
                format(rows=self._rows,
                       cols=self._cols,
                       shape=self._shape))
