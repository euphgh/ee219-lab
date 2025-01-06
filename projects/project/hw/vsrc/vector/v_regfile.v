`include "v_defines.v"

module v_regfile (
    input clk,
    input rst,

    input               vwb_en_i,
    input [`VREG_ADDR_BUS] vwb_addr_i,
    input [`VREG_BUS] vwb_data_i,

    input                    vs1_en_i,
    input      [`VREG_ADDR_BUS] vs1_addr_i,
    output reg [`VREG_BUS] vs1_data_o,

    input                    vs2_en_i,
    input      [`VREG_ADDR_BUS] vs2_addr_i,
    output reg [`VREG_BUS] vs2_data_o
);

  integer i;

  reg [`VREG_BUS] regfile[31:0];

  always @(posedge clk) begin
    if (rst == 1'b1) begin
      for (i = 0; i < 2 ** 32; i = i + 1) begin
        regfile[i] <= {(`VLEN) {1'b0}};
      end
    end else begin
      if ((vwb_en_i == 1'b1) && (vwb_addr_i != 0)) begin
        regfile[vwb_addr_i] <= vwb_data_i;
      end
    end
  end

  always @(*) begin
    if (rst == 1'b1) begin
      vs1_data_o = {(`VLEN) {1'b0}};
    end else begin
      if (vs1_en_i) begin
        vs1_data_o = regfile[vs1_addr_i];
      end else begin
        vs1_data_o = {(`VLEN) {1'b0}};
      end
    end
  end

  always @(*) begin
    if (rst == 1'b1) begin
      vs2_data_o = {(`VLEN) {1'b0}};
    end else begin
      if (vs2_en_i) begin
        vs2_data_o = regfile[vs2_addr_i];
      end else begin
        vs2_data_o = {(`VLEN) {1'b0}};
      end
    end
  end

endmodule

