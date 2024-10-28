import torch
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.Tanh,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features  # Number of input features
        self.out_features = out_features  # Number of output features
        self.grid_size = grid_size  # Grid_size of B-Spline, normally the bigger the better
        self.spline_order = spline_order  # B-Spline order, 3 is recommended, greater computational cost
        """
        1. Calculate the grid step size and generate the grid;
        2. The role of the grid: defining the position of the B-spline basis function;
        3. B Spline basis functions are computed on specific support points, which are defined by a grid. 
           The spline basis function has a specific value and shape at these grid points. 
           Determine the spacing of the spline basis functions;
        4.The grid step (h) determines the distance between grid points, which affects the smoothness and coverage of the spline basis function. 
          The denser the grid, the higher the resolution of the spline basis function and the finer the data can be fitted. 
          Construct the basis used for interpolation and fitting;
        5.Spline basis functions use these grid points to interpolate and can construct continuous and smooth functions.
          Through these basis functions, complex nonlinear transformations of input features can be achieved.
        """
        #grid_step h:
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        # Register the grid as a buffer for the model
        self.register_buffer("grid", grid)  
        
        # Initialize base weight parameters in the shape of (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialize spline weight parameters, shape (out_features, in_features, grid_size + spline_order)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # If the independent scaling spline feature is enabled, initialize the spline scaling parameters with the shape (out_features, in_features)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # Noise scaling factor for adding noise when initializing spline weights
        self.scale_noise = scale_noise

        # Scaling factor for base weights, used when initializing base weights
        self.scale_spline = scale_spline      
        
        # Whether to enable independent spline scaling
        self.enable_standalone_scale_spline = enable_standalone_scale_spline  

        #Basic activation function example, used to perform nonlinear transformation on the input
        self.base_activation = base_activation()

        # Small offset when updating the grid, used to introduce small changes when updating the grid to avoid overfitting
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.base_weight)
        # Add noise to the spline weight parameter spline_weight for initialization
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

        # Calculate the initial value of the spline weight parameter, incorporating the scale_spline scaling factor
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:                
                torch.nn.init.kaiming_normal_(self.spline_scaler)             

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline basis function for a given input tensor.
        Args:
            x (torch.Tensor): input tensor, shape of (batch_size, in_features)。
        Returns:
            torch.Tensor: B-spline basis function tensor with shape (batch_size, in_features, grid_size + spline_order).
        """
        # Make sure the dimension of the input tensor is 2 and its number of columns is equal to the number of input features
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Get grid points (self.grid contained in buffer)
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)        

        # For element-wise operations, extend the last dimension of the input tensor by one dimension
        x = x.unsqueeze(-1)

        # Initialize the basis matrix of the B-spline basis function
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Iteratively calculate spline basis functions
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        # Ensure that the output shape of the B-spline basis function is correct        

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the coefficients of a curve interpolated to a given point.
        These coefficients are used to represent the shape and position of the interpolated curve at a given point.
        Specifically, the method finds the coefficients of the B-spline basis function interpolated at a given point by solving a system of linear equations.
        The function of this method is to calculate the coefficients of the B-spline basis function based on the input and output points,
        This enables these basis functions to accurately interpolate a given input-output point pair.
        This can be used to fit data or apply nonlinear transformations in the model.

        Args:
            x (torch.Tensor): Input points, shape of (batch_size, in_features).
            y (torch.Tensor): Output values corresponding to each input point, shape of (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients of the B-spline basis function, shape of (batch_size, in_features, out_features).
        """    
        # Ensure that the dimension of the input tensor is 2 and that its number of columns is equal to the number of input features
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Ensure that the output tensor is shaped correctly
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # Calculate the B-spline basis function
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)

        # Transpose the output tensor
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)

        # Use linear algebra methods to solve a system of linear equations to find the coefficients
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)

        #Adjust the shape of the result
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # Ensure that the resultant tensor is shaped correctly
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        # Return the continuously stored result tensor
        return result.contiguous()
    
    @property
    def scaled_spline_weight(self):
        """ 
        Compute spline weights with scaling factors.
        spline_weight is a three-dimensional tensor of shape (out_features, in_features, grid_size + spline_order).
        spline_scaler is a two-dimensional tensor with shape (out_features, in_features).
        To enable spline_scaler to be element-wise multiplied by spline_weight,
            The last dimension of spline_scaler needs to be expanded to match the third dimension of spline_weight.
        Returns:
            torch.Tensor: Spline weight tensor with scaling factors.
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )        

    def forward(self, x: torch.Tensor):
        """
        Implement forward propagation of the model.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        # Ensure that the size of the last dimension of the input tensor is equal to the number of input features
        assert x.size(-1) == self.in_features
        
        # Save the original shape of the input tensor
        original_shape = x.shape
        
        # Flatten the input tensor into two dimensions
        x = x.view(-1, self.in_features)

        # Calculate the output of the basic linear transformation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Calculate the output of the B-spline basis function
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # Combine base output and spline output
        output = base_output + spline_output
        
        # Restore the shape of the output tensor
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output
    
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """ 
        The update_grid method is used to dynamically update the grid points of the B-splines according to the input data, thus adapting to the distribution of the input data.
        This method ensures that the B-spline basis function can better fit the data by recalculating and adjusting the grid points.
        This may improve model accuracy and stability during training.

        Args:
            x (torch.Tensor): input tensor,shape of (batch_size, in_features).
            margin (float): Edge size for grid updates, used to introduce small changes when updating the grid.
        """ 
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # arrange (in, batch, coeff)

        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # arrange (in, coeff, out)

        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # arrange (batch, in, out)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))   

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        modified_output = True
    ):
        """
        Initialize KAN model。

        参数:
            layers_hidden (list): List of input and output feature numbers for each layer.
            grid_size (int): Grid size.
            spline_order (int): spline order.
            scale_noise (float): Noise scaling coefficient when initializing spline weights.
            scale_base (float): scaling coefficient when initializing the base weight.
            scale_spline (float): scaling factor when initializing spline weights.
            base_activation (nn.Module): base activation function class.
            grid_eps (float): Small offset when the grid is updated.
            grid_range (list): Grid range.
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size  
        self.spline_order = spline_order  
        self.modified_output = modified_output
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    def forward(self, x: torch.Tensor, update_grid=False):
        
        """
        Args:
            x (torch.Tensor):  (batch_size, in_features)。
            update_grid (bool): use grid update or not.

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        if self.modified_output == False:
            output = x
        elif self.modified_output == True:
            output = F.softplus(x)
        return output         