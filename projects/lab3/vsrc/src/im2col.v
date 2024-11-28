`timescale 1ns / 1ps

`define ZERO_EXT(VALUE, WIDTH) {{(WIDTH-$bits(VALUE)){1'b0}}, VALUE}

module im2col #(
    parameter IMG_C         = 1,
    parameter IMG_W         = 8,
    parameter IMG_H         = 8,
    parameter DATA_WIDTH    = 8,
    parameter ADDR_WIDTH    = 32,
    parameter FILTER_SIZE   = 3,
    parameter IMG_BASE      = 16'h0000,
    parameter IM2COL_BASE   = 16'h2000
) (
    input   clk,
    input   rst_n,
    input   [DATA_WIDTH-1:0] data_rd,
    output  [DATA_WIDTH-1:0] data_wr,
    output  [ADDR_WIDTH-1:0] addr_wr,
    output  [ADDR_WIDTH-1:0] addr_rd,
    output  reg done,
    output  mem_wr_en
);

localparam PADDING = FILTER_SIZE == 1 ? 32'd0 : 32'd1;
localparam OUT_H = (IMG_H + 2 * PADDING - FILTER_SIZE) + 1;
localparam OUT_W = (IMG_W + 2 * PADDING - FILTER_SIZE) + 1;
localparam GEMM_W = FILTER_SIZE * FILTER_SIZE * IMG_C;
localparam GEMM_H = OUT_H * OUT_W;

/* nested counters */
reg [ADDR_WIDTH-1:0] counters [0:4];
wire [ADDR_WIDTH-1:0] cnt_next [0:4];
wire is_max [0:4];
/* verilator lint_off UNOPTFLAT */
wire wrap_next [0:4];
reg tick;
localparam [ADDR_WIDTH-1:0] MAX_VALUES [0:4] = {FILTER_SIZE, FILTER_SIZE, IMG_C, OUT_W, OUT_H};

generate
    for (genvar i = 0; i < 5; i = i + 1) begin
        assign is_max[i] = counters[i] == MAX_VALUES[i] - 1;
        assign cnt_next[i] = is_max[i] ? 0 : counters[i] + 1;
    end
endgenerate

always @(posedge clk) begin
    if (!rst_n) begin
        tick <= 0;
    end
    else begin
        tick <= ~tick;
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        counters[0] <= 0;
    end
    else if (tick) begin
        counters[0] <= cnt_next[0]; 
    end
end
assign wrap_next[0] = is_max[0];

generate
    for (genvar i = 1; i < 5; i = i + 1) begin
        always @(posedge clk) begin
            if (!rst_n || done) begin
                counters[i] <= 0;
            end
            else begin
                if (wrap_next[i-1] && tick) begin
                    counters[i] <= cnt_next[i];
                end
            end
        end
        assign wrap_next[i] = wrap_next[i-1] && is_max[i];
    end
endgenerate

wire [ADDR_WIDTH-1:0] h = counters[4];
wire [ADDR_WIDTH-1:0] w = counters[3];
wire [ADDR_WIDTH-1:0] c = counters[2];
wire [ADDR_WIDTH-1:0] row = counters[1];
wire [ADDR_WIDTH-1:0] col = counters[0];

wire [ADDR_WIDTH-1:0] x_idx = w + h * OUT_W;
wire [ADDR_WIDTH-1:0] y_idx = col + row * FILTER_SIZE + c * (FILTER_SIZE * FILTER_SIZE);


assign addr_rd = c + ((w + col - PADDING) * IMG_C) + ((h + row - PADDING) * (IMG_C * IMG_W)) +  `ZERO_EXT(IMG_BASE, ADDR_WIDTH);

/* column major */
assign addr_wr = x_idx + (y_idx * GEMM_H) + `ZERO_EXT(IM2COL_BASE, ADDR_WIDTH);

// becasue padding is may be zero
/* verilator lint_off UNSIGNED */
wire x_is_padding = !(PADDING <= (h + row) && (h + row) <= IMG_H + PADDING -1);
/* verilator lint_off UNSIGNED */
wire y_is_padding = !(PADDING <= (w + col) && (w + col) <= IMG_W + PADDING -1);

wire is_padding = x_is_padding || y_is_padding;

reg [DATA_WIDTH-1:0] data_reg;

reg [1:0] state;
localparam LOAD = 0;
localparam STORE = 1;
localparam DONE = 2;
assign mem_wr_en = state == STORE;
assign data_wr = is_padding ? 'd0 : data_rd;

always @(posedge clk) begin
    if (!rst_n) begin
        done <= 0;
    end
    else begin
        done <= (x_idx == GEMM_H - 1) && (y_idx == GEMM_W - 1) && mem_wr_en;
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        state       <= LOAD; 
        data_reg    <= 0;
    end
    else if (state == LOAD) begin
        state       <= done ? DONE : STORE;
    end
    else if (state == STORE) begin
        state       <= LOAD;
    end
end


endmodule