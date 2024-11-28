`timescale 1ns / 1ps

module Repeater #(
    parameter cycleNR = 0,
    parameter DATA_WIDTH = 32
) (
    input clk,
    input rst_n,
    input [DATA_WIDTH-1:0] data_in,
    output [DATA_WIDTH-1:0] data_out
);

generate
    if (cycleNR == 0) begin
        assign data_out = data_in;
    end
    else begin
        reg [DATA_WIDTH-1:0] stage [cycleNR-1:0];
        always @(posedge clk) begin
                if (!rst_n) begin
                    stage[0] <= 0;
                end
                else begin
                    stage[0] <= data_in;
                end
        end
        for (genvar i = 1; i < cycleNR; i = i + 1) begin
            always @(posedge clk) begin
                if (!rst_n) begin
                    stage[i] <= 0;
                end
                else begin
                    stage[i] <= stage[i-1];
                end
            end
        end
        assign data_out = stage[cycleNR-1];
    end
endgenerate
    
endmodule

module systolic_array#(
    parameter M           = 5,
    parameter N           = 3,
    parameter K           = 4,
    parameter DATA_WIDTH  = 32
) (
    input   clk,
    input   rst_n,
    input   [DATA_WIDTH*M-1:0] X,
    input   [DATA_WIDTH*K-1:0] W,
    output  [DATA_WIDTH*M*K-1:0] Y,
    output  reg done
);

    wire [DATA_WIDTH-1:0] x_in [M-1:0][K:0];
    wire [DATA_WIDTH-1:0] w_in [M:0][K-1:0];

generate
    for (genvar m = 0; m < M; m = m + 1) begin
        wire [DATA_WIDTH-1:0] data_in  = X[(m + 1) * DATA_WIDTH - 1: m * DATA_WIDTH];
        wire [DATA_WIDTH-1:0] data_out;
        assign x_in[m][0] = data_out;
        Repeater #(
            .cycleNR(m),
            .DATA_WIDTH(DATA_WIDTH)
        ) rep_x (
            .clk(clk),
            .rst_n(rst_n),
            .data_in(data_in),
            .data_out(data_out)
        );
    end
    for (genvar k = 0; k < K; k = k + 1) begin
        wire [DATA_WIDTH-1:0] data_in  = W[(k + 1) * DATA_WIDTH - 1: k * DATA_WIDTH];
        wire [DATA_WIDTH-1:0] data_out;
        assign w_in[0][k] = data_out;
        Repeater #(
            .cycleNR(k),
            .DATA_WIDTH(DATA_WIDTH)
        ) rep_w (
            .clk(clk),
            .rst_n(rst_n),
            .data_in(data_in),
            .data_out(data_out)
        );
    end
endgenerate

generate
    for (genvar m = 0; m < M; m = m + 1) begin
        for (genvar k = 0; k < K; k = k + 1) begin
            wire [DATA_WIDTH-1:0] y_out;
            pe#(
                .DATA_WIDTH(DATA_WIDTH)
            ) pe (
                .clk(clk),
                .rst(rst_n),
                .x_in   (x_in[m  ][k  ]),
                .w_in   (w_in[m  ][k  ]),
                .x_out  (x_in[m  ][k+1]),
                .w_out  (w_in[m+1][k  ]),
                .y_out  (y_out)
            );
            assign Y[m*K*DATA_WIDTH+(k+1)*DATA_WIDTH-1:m*K*DATA_WIDTH+k*DATA_WIDTH] = y_out;
        end
    end
endgenerate

reg [DATA_WIDTH-1:0] cnt;
always @(posedge clk) begin
    if (!rst_n) begin
        cnt <= 0;
        done <= 0;
    end
    else begin
        cnt <= cnt + 1;
        done <= (cnt == M + K + N - 1);
    end
end

endmodule