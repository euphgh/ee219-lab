`timescale 1ns / 1ps

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
    input clk,
    input rst_n,
    input [DATA_WIDTH-1:0] data_rd,
    output reg [DATA_WIDTH-1:0] data_wr,
    output reg [ADDR_WIDTH-1:0] addr_wr,
    output reg [ADDR_WIDTH-1:0] addr_rd,
    output reg done,
    output reg mem_wr_en
);

localparam PADDED_H = IMG_H + 2*(FILTER_SIZE/2);
localparam PADDED_W = IMG_W + 2*(FILTER_SIZE/2);
localparam OUT_H = IMG_H;
localparam OUT_W = IMG_W;

reg [DATA_WIDTH-1:0] im2col_mat [0:OUT_H*OUT_W-1][0:FILTER_SIZE*FILTER_SIZE*IMG_C-1];
reg [DATA_WIDTH-1:0] padded_img [0:IMG_C-1][0:PADDED_H-1][0:PADDED_W-1];
reg [DATA_WIDTH-1:0] img [0:IMG_C-1][0:IMG_H-1][0:IMG_W-1];

reg [31 : 0] rd_cnt, wr_cnt;

localparam IDLE = 3'd0,
           READ = 3'd1,
           PADDING = 3'd2,
           CONVERT = 3'd3,
           DONE = 3'd4;

reg [2:0] state;


always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        addr_rd <= IMG_BASE;
        addr_wr <= IM2COL_BASE;
        rd_cnt <= 0;
        wr_cnt <= 0;
    end else begin
        case (state)
            IDLE: begin
                addr_rd <= IMG_BASE + 1;
                rd_cnt <= 0;
                state <= READ;
            end 
            READ: begin
                if (rd_cnt < IMG_C*IMG_H*IMG_W) begin
                    addr_rd <= addr_rd + 1;
                    rd_cnt <= rd_cnt + 1;
                    img[rd_cnt % IMG_C][rd_cnt / (IMG_C*IMG_W)][rd_cnt / IMG_C % IMG_W] <= data_rd;
                end else begin
                    state <= PADDING;
                end
            end
            PADDING: begin
                for (int c = 0; c < IMG_C; c++) begin
                    for (int h = 0; h < PADDED_H; h = h + 1) begin
                        for (int w = 0; w < PADDED_W; w = w + 1) begin
                            if (h < FILTER_SIZE / 2 || h >= IMG_H + FILTER_SIZE / 2 || w < FILTER_SIZE / 2 || w >= IMG_W + FILTER_SIZE / 2) begin
                                padded_img[c][h][w] = 0;
                            end else begin
                                padded_img[c][h][w] = img[c][h - FILTER_SIZE / 2][w - FILTER_SIZE / 2];
                            end
                        end
                    end
                end
                state <= CONVERT;
            end
            CONVERT: begin
                for (int h = 0; h < OUT_H; h = h + 1) begin
                    for (int w = 0; w < OUT_W; w = w + 1) begin
                        for (int c = 0; c < IMG_C; c = c + 1) begin
                            for (int fh = 0; fh < FILTER_SIZE; fh = fh + 1) begin
                                for (int fw = 0; fw < FILTER_SIZE; fw = fw + 1) begin
                                    im2col_mat[h*OUT_W + w][c*FILTER_SIZE*FILTER_SIZE + fh*FILTER_SIZE + fw] = padded_img[c][h + fh][w + fw];
                                end
                            end
                        end
                    end
                end

                addr_wr <= IM2COL_BASE-1;
                wr_cnt <= 0;
                state <= DONE;
            end
            DONE: begin
                if (wr_cnt < OUT_H*OUT_W*FILTER_SIZE*FILTER_SIZE*IMG_C) begin
                    addr_wr <= addr_wr + 1;
                    wr_cnt <= wr_cnt + 1;
                    data_wr <= im2col_mat[wr_cnt % (OUT_H*OUT_W)][wr_cnt / (OUT_H*OUT_W)];
                    mem_wr_en <= 1;
                    state <= DONE;
                end else begin
                    done <= 1;
                    state <= IDLE;
                    mem_wr_en <= 0;
                end
            end
            default: begin
                state <= IDLE;
            end

        endcase
    end
end


endmodule