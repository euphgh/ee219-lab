`timescale 1ns / 1ps

module systolic_array#(
    parameter M           = 5,
    parameter N           = 3,
    parameter K           = 4,
    parameter DATA_WIDTH  = 32
) (
    input clk,
    input rst_n,
    input [DATA_WIDTH*M-1:0] X,
    input [DATA_WIDTH*K-1:0] W,
    output reg [DATA_WIDTH*M*K-1:0] Y,
    output reg done
);

wire [DATA_WIDTH-1:0] x_in [M-1:0][K:0];  
wire [DATA_WIDTH-1:0] w_in [M:0][K-1:0];  
wire [DATA_WIDTH-1:0] pe_out [M-1:0][K-1:0];

reg [DATA_WIDTH-1:0] x_delay [M-1:0][M-1:0];
reg [DATA_WIDTH-1:0] w_delay [K-1:0][K-1:0];

reg [31:0] counter;

genvar i, j;
generate
    for(i = 0; i < M; i = i + 1) begin
        for(j = 0; j < K; j = j + 1) begin
            pe #(
                .DATA_WIDTH(DATA_WIDTH)
            ) pe_inst (
                .clk(clk),
                .rst(!rst_n),
                .x_in(x_in[i][j]),
                .w_in(w_in[i][j]),
                .x_out(x_in[i][j+1]),
                .w_out(w_in[i+1][j]),
                .y_out(pe_out[i][j])
            );
        end
    end
endgenerate

integer m, n;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for(m = 0; m < M; m = m + 1) begin
            for(n = 0; n < M; n = n + 1) begin
                x_delay[m][n] <= 0;
            end
        end
    end else begin
        for(m = 0; m < M; m = m + 1) begin
            x_delay[m][0] <= X[DATA_WIDTH*(m+1)-1 -: DATA_WIDTH];
            for(n = 1; n < M; n = n + 1) begin
                x_delay[m][n] <= x_delay[m][n-1];
            end
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for(m = 0; m < K; m = m + 1) begin
            for(n = 0; n < K; n = n + 1) begin
                w_delay[m][n] <= 0;
            end
        end
    end else begin
        for(m = 0; m < K; m = m + 1) begin
            w_delay[m][0] <= W[DATA_WIDTH*(m+1)-1 -: DATA_WIDTH];
            for(n = 1; n < K; n = n + 1) begin
                w_delay[m][n] <= w_delay[m][n-1];
            end
        end
    end
end

generate
    for(i = 0; i < M; i = i + 1) begin
        assign x_in[i][0] = x_delay[i][i];
    end
    for(j = 0; j < K; j = j + 1) begin
        assign w_in[0][j] = w_delay[j][j];
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        counter <= 0;
        done <= 0;
    end else begin
        if (counter < N + M + K - 1) begin
            counter <= counter + 1;
            done <= 0;
        end else begin
            done <= 1;
        end
    end
end

always @(*) begin
    for(m = 0; m < M; m = m + 1) begin
        for(n = 0; n < K; n = n + 1) begin
            Y[DATA_WIDTH*(m*K+n+1)-1 -: DATA_WIDTH] = pe_out[m][n];
        end
    end
end

endmodule