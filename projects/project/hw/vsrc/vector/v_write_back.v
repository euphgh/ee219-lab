`include "v_defines.v"

module v_write_back (
    input clk,
    input rst,

    input                  vid_wb_en_i,
    input                  vid_wb_sel_i,
    input [`VREG_ADDR_BUS] vid_wb_addr_i,
    input [     `VREG_BUS] valu_result_i,
    input [     `VREG_BUS] vmem_result_i,

    output                  vwb_en_o,
    output [`VREG_ADDR_BUS] vwb_addr_o,
    output [     `VREG_BUS] vwb_data_o
);

  assign vwb_en_o   = vid_wb_en_i;
  assign vwb_addr_o = vid_wb_addr_i;
  assign vwb_data_o = (vid_wb_sel_i) ? vmem_result_i : valu_result_i;

endmodule

