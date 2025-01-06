`include "v_defines.v"

module v_mem (
    input                   clk,
    input                   rst,
    
    input                   vmem_ren_i,
    input                   vmem_wen_i,
    input   [`VMEM_ADDR_BUS]   vmem_addr_i,
    input   [`VMEM_DATA_BUS]   vram_w_data,
    output  [`VMEM_DATA_BUS]   vram_r_data,

    output                  vram_ren_o,
    output                  vram_wen_o,
    output  [`VMEM_ADDR_BUS]   vram_addr_o,
    output  [`VMEM_DATA_BUS]   vram_mask_o
);

assign vram_ren_o   = vmem_ren_i ;
assign vram_wen_o   = vmem_wen_i ;
assign vram_addr_o  = vmem_addr_i ;
assign vram_din_o   = vmem_din_i ;
assign vram_mask_o  = {(`VLEN){1'b1}};
assign vmem_dout_o  = vram_dout_i ;

endmodule

